import os
import time
import copy
import asyncio
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils import data
from typing import List, Dict, Optional
import logging
import traceback
import sys
from datetime import datetime
import h5py
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix

# 导入MolTrans模型和工具
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

# 配置日志系统
def setup_training_logging():
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 获取训练专用日志器
    logger = logging.getLogger("TrainingManager")
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器，每天一个日志文件
        log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def get_task(task_name):
    """Get dataset path"""
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'

class TrainingManager:
    def __init__(self, active_connections: List):
        """Initialize the training manager"""
        self.logger = logging.getLogger('training_manager')
        self.active_connections = active_connections
        
        # 训练状态
        self.status = "idle"  # idle, training, completed, error
        self.progress = 0
        self.start_time = None
        self.end_time = None
        
        # 当前训练指标
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.metrics_history = []
        self.current_metrics = {}
        
        # 可视化和模型
        self.plots = {}
        self.best_model = None
        self.best_threshold = None
        self.confusion_matrix = None  # 新增: 用于存储混淆矩阵
        
        # 训练配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f"TrainingManager initialized, using device: {self.device}")
        
        # 计算资源信息
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        
    async def train_model(self, config):
        """Start the model training process"""
        self.status = "preparing"
        self.progress = 0
        self.metrics_history = []
        self.start_time = time.time()
        self.cancel_training = False
        
        self.logger.info(f"Starting model training, config: {config}")
        self.logger.info(f"Training device: {self.device}")
        
        await self._notify_progress()
        
        # 设置训练参数
        self.total_epochs = config.epochs
        dataset = config.dataset
        batch_size = config.batch_size
        lr = config.learning_rate
        
        self.logger.info(f"Training parameters - Dataset: {dataset}, Batch Size: {batch_size}, Learning Rate: {lr}, Total Epochs: {self.total_epochs}")
        
        # 加载数据
        self.status = "loading_data"
        self.progress = 5
        await self._notify_progress()
        
        try:
            # 准备数据加载器
            dataFolder = get_task(dataset)
            self.logger.info(f"Loading dataset {dataset}, path: {dataFolder}")
            
            try:
                df_train = pd.read_csv(os.path.join(dataFolder, 'train.csv'))
                df_val = pd.read_csv(os.path.join(dataFolder, 'val.csv'))
                df_test = pd.read_csv(os.path.join(dataFolder, 'test.csv'))
                
                self.logger.info(f"Training samples: {len(df_train)}, Validation samples: {len(df_val)}, Test samples: {len(df_test)}")
                self.logger.debug(f"Positive ratio - Train: {df_train.Label.mean():.3f}, Val: {df_val.Label.mean():.3f}, Test: {df_test.Label.mean():.3f}")
                
                params = {'batch_size': batch_size, 'shuffle': True,
                        'num_workers': 4, 'drop_last': True}
                test_params = {'batch_size': batch_size, 'shuffle': False,
                               'num_workers': 4, 'drop_last': False}
                
                self.logger.info(f"Creating data loaders, Batch Size: {batch_size}, Num Workers: 4")
                
                training_generator = data.DataLoader(
                    BIN_Data_Encoder(df_train.index, df_train.Label, df_train),**params)
                validation_generator = data.DataLoader(
                    BIN_Data_Encoder(df_val.index, df_val.Label, df_val),**params)
                test_generator = data.DataLoader(
                    BIN_Data_Encoder(df_test.index, df_test.Label, df_test),**test_params)
                
                self.total_batches = len(training_generator)
                self.logger.info(f"Data loaders created, batches per training epoch: {self.total_batches}")
                
            except Exception as e:
                self.logger.error(f"Data loading failed: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise
            
            # 初始化模型
            self.status = "initializing_model"
            self.progress = 10
            await self._notify_progress()
            
            try:
                self.logger.info("Initializing model...")
                model_config = BIN_config_DBPE()
                model_config['batch_size'] = batch_size
                self.logger.debug(f"Model config: {model_config}")
                
                model = BIN_Interaction_Flat(**model_config).to(self.device)
                self.logger.info(f"Model created: {model.__class__.__name__}")
                
                # 计算模型参数
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.logger.info(f"Model total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
                
            except Exception as e:
                self.logger.error(f"Model initialization failed: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise
            
            # 优化器设置
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            self.logger.info(f"Optimizer setup complete: Adam, Learning Rate: {lr}")
            
            # 训练循环
            self.status = "training"
            max_auc = 0
            best_threshold = 0.5
            model_max = copy.deepcopy(model)
            
            self.logger.info(f"Starting training loop, total epochs: {self.total_epochs}")
            
            for epoch in range(self.total_epochs):
                # 检查是否取消
                if self.cancel_training:
                    self.logger.warning(f"Training cancelled by user before epoch {epoch+1}/{self.total_epochs}")
                    self.status = "cancelled"
                    self.progress = min(self.progress, 99)
                    await self._notify_progress(message="Training cancelled by user")
                    return {"status": "cancelled", "message": "Training was cancelled by user"}
                
                self.current_epoch = epoch + 1
                self.current_batch = 0
                epoch_loss = 0.0
                batch_count = 0
                epoch_start_time = time.time()
                
                self.logger.info(f"Starting epoch {self.current_epoch}/{self.total_epochs} training")
                
                # 训练阶段
                model.train()
                # 添加用于收集训练集预测和标签的列表
                train_preds = torch.Tensor([])
                train_labels = torch.Tensor([])
                
                for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
                    batch_start_time = time.time()
                    # 检查是否取消
                    if self.cancel_training:
                        self.logger.warning(f"Training cancelled by user during epoch {self.current_epoch}/{self.total_epochs}, batch {i+1}/{self.total_batches}")
                        self.status = "cancelled"
                        self.progress = min(self.progress, 99)
                        await self._notify_progress(message_text=f"Training cancelled by user at epoch {self.current_epoch}, batch {self.current_batch}")
                        return {"status": "cancelled", "message": f"Training was cancelled during epoch {self.current_epoch}, batch {self.current_batch}"}
                    
                    self.current_batch = i + 1
                    progress_percent = (epoch / self.total_epochs) * 100
                    batch_progress = (i / len(training_generator)) * (100 / self.total_epochs)
                    self.progress = min(90, int(progress_percent + batch_progress))
                    
                    # 定期更新进度
                    if i % 10 == 0:
                        await self._notify_progress()
                        self.logger.debug(f"Epoch {self.current_epoch}/{self.total_epochs}, Batch {i+1}/{self.total_batches}, Progress: {self.progress}%")
                    
                    # 训练步骤
                    d = d.long().to(self.device)
                    p = p.long().to(self.device)
                    d_mask = d_mask.long().to(self.device)
                    p_mask = p_mask.long().to(self.device)
                    label_tensor = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)
                    
                    optimizer.zero_grad()
                    score = model(d, p, d_mask, p_mask)
                    m = torch.nn.Sigmoid()
                    logits = torch.squeeze(m(score))
                    loss = torch.nn.BCELoss()(logits, label_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    batch_end_time = time.time()
                    batch_duration = batch_end_time - batch_start_time
                    
                    # 收集训练批次的预测和标签以计算epoch指标
                    train_preds = torch.cat((train_preds, logits.cpu().detach()), 0)
                    train_labels = torch.cat((train_labels, label_tensor.cpu().detach()), 0)
                    
                    # 每20个批次记录一次详细信息
                    if i % 20 == 0:
                        self.logger.debug(f"Batch {i+1}/{self.total_batches} - Loss: {loss.item():.4f}, Batch time: {batch_duration:.2f}s")
                    
                    # 检查每个批次结束时是否有取消请求
                    if self.cancel_training:
                        self.logger.warning(f"Training cancelled by user after epoch {self.current_epoch}/{self.total_epochs}, batch {i+1}/{self.total_batches}")
                        self.status = "cancelled"
                        self.progress = min(self.progress, 99)
                        await self._notify_progress(message_text=f"Training cancelled by user at epoch {self.current_epoch}, batch {self.current_batch}")
                        return {"status": "cancelled", "message": f"Training was cancelled after epoch {self.current_epoch}, batch {self.current_batch}"}
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                avg_epoch_loss = epoch_loss / batch_count
                
                # 计算训练集的性能指标
                train_label_arr = train_labels.numpy()
                train_pred_arr = train_preds.numpy()
                
                try:
                    train_auc = roc_auc_score(train_label_arr, train_pred_arr)
                    train_auprc = average_precision_score(train_label_arr, train_pred_arr)
                    # 使用验证集的最佳阈值来计算训练集F1，或使用0.5默认值
                    train_f1_threshold = best_threshold if best_threshold else 0.5
                    train_pred_label = (train_pred_arr >= train_f1_threshold).astype(int)
                    train_f1 = f1_score(train_label_arr, train_pred_label)
                    self.logger.info(f"Epoch {self.current_epoch}/{self.total_epochs} training finished - Avg Loss: {avg_epoch_loss:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f}, F1: {train_f1:.4f}, Duration: {epoch_duration:.2f}s")
                except Exception as e:
                    self.logger.error(f"Error calculating training set metrics: {str(e)}")
                    train_auc, train_auprc, train_f1 = 0.0, 0.0, 0.0
                
                # 验证阶段
                val_start_time = time.time()
                self.logger.info(f"Starting epoch {self.current_epoch}/{self.total_epochs} validation")
                
                auc_val, auprc_val, f1_val, _, val_loss, thred_optim, _ = await self._test(
                    validation_generator, model)
                
                val_end_time = time.time()
                val_duration = val_end_time - val_start_time
                
                self.logger.info(f"Epoch {self.current_epoch}/{self.total_epochs} validation finished - AUC: {auc_val:.4f}, AUPRC: {auprc_val:.4f}, F1: {f1_val:.4f}, Loss: {val_loss:.4f}, Best Threshold: {thred_optim:.4f}, Duration: {val_duration:.2f}s")
                
                # 记录指标 (包括训练指标)
                metrics = {
                    "epoch": self.current_epoch,
                    "train_loss": avg_epoch_loss,
                    "train_auc": train_auc,
                    "train_auprc": train_auprc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_auc": auc_val,
                    "val_auprc": auprc_val,
                    "val_f1": f1_val,
                    "learning_rate": lr
                }
                self.metrics_history.append(metrics)
                self.current_metrics = metrics
                
                self.logger.debug(f"Epoch {self.current_epoch} metrics: {metrics}")
                
                # 更新最佳模型
                if auc_val > max_auc:
                    old_max_auc = max_auc
                    max_auc = auc_val
                    model_max = copy.deepcopy(model)
                    best_threshold = thred_optim
                    self.logger.info(f"New best model found - Validation AUC improved: {old_max_auc:.4f} -> {max_auc:.4f}")
                
                # 生成训练曲线
                plot_start_time = time.time()
                await self._generate_training_plots()
                plot_end_time = time.time()
                self.logger.debug(f"Training plots generated, duration: {plot_end_time - plot_start_time:.2f}s")
                
                # 更新进度
                await self._notify_progress()
            
            # 训练完成
            self.best_model = model_max
            self.status = "completed"
            self.progress = 100
            
            training_duration = time.time() - self.start_time
            self.logger.info(f"Training completed! Total {self.total_epochs} epochs, total duration: {training_duration:.2f}s")
            self.logger.info(f"Best validation performance: AUC={max_auc:.4f}, Threshold={best_threshold:.4f}")
            
            # 使用最佳模型再次评估验证集 (确保混淆矩阵准确)
            self.logger.info("Re-evaluating validation set with the best model...")
            final_val_auc, final_val_auprc, final_val_f1, _, final_val_loss, final_val_threshold, _ = await self._test(
                validation_generator, self.best_model)
            self.logger.info(f"Best model validation results - AUC: {final_val_auc:.4f}, AUPRC: {final_val_auprc:.4f}, F1: {final_val_f1:.4f}, Loss: {final_val_loss:.4f}")
            
            # 使用最佳模型评估测试集
            self.logger.info("Evaluating test set with the best model...")
            try:
                test_auc, test_auprc, test_f1, _, test_loss, _, _ = await self._test(
                    test_generator, self.best_model)
                self.logger.info(f"Best model test results - AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}, Loss: {test_loss:.4f}")
                # 存储测试结果以供绘图使用
                self.test_metrics = {
                    "test_loss": test_loss,
                    "test_auc": test_auc,
                    "test_auprc": test_auprc,
                    "test_f1": test_f1
                }
            except Exception as e:
                self.logger.error(f"Error evaluating test set: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.test_metrics = None # Indicate test failed
            
            # 通知前端 (只报告验证集性能)
            await self._notify_progress(message_text=f"Training complete, best validation performance - AUC: {final_val_auc:.4f}, F1: {final_val_f1:.4f}")
            
            self.end_time = time.time()
            self.best_threshold = best_threshold
            
            # 生成训练曲线和热力图 (现在会包含测试集数据点)
            await self._generate_training_plots()
            
            # 保存模型
            if config.save_model:
                save_start_time = time.time()
                model_path = await self._save_model(dataset, best_threshold)
                save_end_time = time.time()
                self.logger.info(f"Model saving complete, path: {model_path}, duration: {save_end_time - save_start_time:.2f}s")
            
            # 最终进度更新
            await self._notify_progress(message_text="Training and evaluation fully completed")
            
            # 返回结果
            return {
                "status": "success",
                "auc": max_auc,
                "best_threshold": best_threshold,
                "best_epoch": self.current_epoch,
                "total_time": time.time() - self.start_time,
                "metrics_history": self.metrics_history,
                "plots": self.plots
            }
        except Exception as e:
            self.status = "error"
            self.progress = 0
            
            # 记录错误详情
            self.logger.error(f"Error occurred during training: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # 通知前端
            await self._notify_progress(error=str(e))
            
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _test(self, data_generator, model):
        """Test/validate model performance"""
        test_start = time.time()
        self.logger.info(f"_test: Function started. Data generator length: {len(data_generator)}")
        model.eval()
        
        y_pred = torch.Tensor([])
        y_label = torch.Tensor([])
        losses = []
        
        self.logger.debug("_test: Starting evaluation loop...")
        
        try:
            with torch.no_grad():
                for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
                    batch_start_time = time.time()
                    self.logger.debug(f"_test: Processing batch {i+1}/{len(data_generator)}")
                    
                    # Move data to device
                    try:
                        d = d.long().to(self.device)
                        p = p.long().to(self.device)
                        d_mask = d_mask.long().to(self.device)
                        p_mask = p_mask.long().to(self.device)
                        label_tensor = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)
                        self.logger.debug(f"_test: Batch {i+1} data moved to {self.device}")
                    except Exception as e_data:
                        self.logger.error(f"_test: Error moving data for batch {i+1} to device: {e_data}")
                        # Optionally skip this batch or raise error
                        continue 

                    # Forward pass
                    try:
                        score = model(d, p, d_mask, p_mask)
                        m = torch.nn.Sigmoid()
                        logits = torch.squeeze(m(score))
                        loss = torch.nn.BCELoss()(logits, label_tensor)
                        self.logger.debug(f"_test: Batch {i+1} forward pass completed. Loss: {loss.item():.4f}")
                    except Exception as e_fwd:
                        self.logger.error(f"_test: Error during forward pass for batch {i+1}: {e_fwd}")
                        self.logger.error(traceback.format_exc())
                        # Optionally skip or raise
                        continue

                    # Collect results
                    try:
                        losses.append(loss.item())
                        y_label = torch.cat((y_label, label_tensor.cpu()), 0)
                        y_pred = torch.cat((y_pred, logits.cpu()), 0)
                        batch_end_time = time.time()
                        self.logger.debug(f"_test: Batch {i+1} results collected. Batch time: {batch_end_time - batch_start_time:.2f}s")
                    except Exception as e_coll:
                        self.logger.error(f"_test: Error collecting results for batch {i+1}: {e_coll}")
                        # Optionally skip or raise
                        continue
                        
            self.logger.info(f"_test: Evaluation loop finished after {time.time() - test_start:.2f} seconds.")
            
            # --- Metric Calculation (Add similar try-except blocks if needed) ---
            self.logger.debug("_test: Starting metric calculation...")
            label_arr = y_label.cpu().detach().numpy()
            pred_arr = y_pred.cpu().detach().numpy()
            
            self.logger.debug(f"_test: Predicted values stats: Min={pred_arr.min():.4f}, Max={pred_arr.max():.4f}, Mean={pred_arr.mean():.4f}, Std={pred_arr.std():.4f}")
            self.logger.debug(f"_test: Label values stats: Positives={np.sum(label_arr)}, Total={len(label_arr)}, Positive Ratio={np.mean(label_arr):.4f}")
            
            # 计算AUC
            try:
                auc_test = roc_auc_score(label_arr, pred_arr)
                self.logger.debug(f"_test: AUC calculated successfully: {auc_test:.4f}")
            except Exception as e:
                self.logger.error(f"_test: Error calculating AUC: {str(e)}")
                auc_test = 0.5  # Default value
            
            # 计算AUPRC
            try:
                auprc_test = average_precision_score(label_arr, pred_arr)
                self.logger.debug(f"_test: AUPRC calculated successfully: {auprc_test:.4f}")
            except Exception as e:
                self.logger.error(f"_test: Error calculating AUPRC: {str(e)}")
                auprc_test = 0.0  # Default value
            
            # 找到最佳阈值
            try:
                fpr, tpr, thresholds = roc_curve(label_arr, pred_arr)
                self.logger.debug(f"_test: ROC curve calculated successfully. Thresholds found: {len(thresholds)}")
                
                # 安全处理除法，避免除以零错误
                with np.errstate(divide='raise', invalid='raise'):
                    try:
                        precision = np.divide(tpr, (fpr + tpr), out=np.zeros_like(tpr), where=(fpr + tpr)!=0)
                        self.logger.debug(f"_test: Precision calculated successfully")
                        
                        f1 = np.divide(2 * precision * tpr, (precision + tpr), out=np.zeros_like(tpr), where=(precision + tpr)!=0)
                        self.logger.debug(f"_test: F1 calculated successfully")
                    except FloatingPointError as e:
                        self.logger.error(f"_test: Floating point error during F1 calculation: {str(e)}")
                        # 回退到更安全的计算方法
                        precision = np.zeros_like(tpr)
                        mask = (fpr + tpr) > 0
                        precision[mask] = tpr[mask] / (fpr[mask] + tpr[mask])
                        
                        f1 = np.zeros_like(tpr)
                        mask = (precision + tpr) > 0
                        f1[mask] = 2 * precision[mask] * tpr[mask] / (precision[mask] + tpr[mask])
                        self.logger.debug("Recalculated F1 using safe division method successfully")
                
                # 处理nan值
                f1 = np.nan_to_num(f1)
                self.logger.debug(f"_test: F1 values stats: Min={np.min(f1):.4f}, Max={np.max(f1):.4f}, Mean={np.mean(f1):.4f}")
                
                # 找到F1最大的索引和对应的阈值
                best_idx = np.argmax(f1)
                thred_optim = thresholds[best_idx]
                self.logger.debug(f"_test: Optimal threshold found: {thred_optim:.4f} (corresponding F1: {f1[best_idx]:.4f})")
            except Exception as e:
                self.logger.error(f"_test: Error calculating optimal threshold: {str(e)}")
                thred_optim = 0.5  # Default threshold
                f1 = np.array([0.0])
            
            # 使用最佳阈值计算最终的F1
            try:
                pred_label = (pred_arr >= thred_optim).astype(int)
                f1_test = f1_score(label_arr, pred_label)
                self.logger.debug(f"_test: Final F1 calculated using threshold {thred_optim:.4f}: {f1_test:.4f}")
                
                # 计算混淆矩阵
                cm = confusion_matrix(label_arr, pred_label)
                self.confusion_matrix = cm
                self.logger.debug(f"_test: Confusion matrix calculated successfully:\n{cm}")
                
                # 打印TP, FP, TN, FN
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    self.logger.debug(f"_test: CM values - TP={tp}, FP={fp}, TN={tn}, FN={fn}")
                    self.logger.debug(f"_test: Accuracy - Positive: {tp/(tp+fp) if tp+fp>0 else 0:.4f}, Negative: {tn/(tn+fn) if tn+fn>0 else 0:.4f}")
            
            except Exception as e:
                self.logger.error(f"_test: Error calculating final F1 or confusion matrix: {str(e)}")
                self.logger.error(traceback.format_exc())
                f1_test = 0.0
                self.confusion_matrix = np.array([[0, 0], [0, 0]])
            
            # 计算平均损失
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            self.logger.info(f"_test: Function finished successfully. Total duration: {time.time() - test_start:.2f}s")
            return auc_test, auprc_test, f1_test, pred_label, avg_loss, thred_optim, f1
        except Exception as e_loop:
            self.logger.error(f"_test: Error during evaluation loop or metric calculation: {e_loop}")
            self.logger.error(traceback.format_exc())
            # Return default/error values to prevent crash in calling function
            return 0.5, 0.0, 0.0, np.array([]), 0.0, 0.5, np.array([0.0])
    
    async def _generate_training_plots(self):
        """Generate training curve visualizations and confusion matrix heatmap"""
        if not self.metrics_history:
            self.logger.warning("Cannot generate training plots, metrics_history is empty")
            return
        
        self.logger.debug("Starting generation of training plots and heatmap...")
        
        try:
            # 确保目录存在
            save_dir = "static/images"
            try:
                os.makedirs(save_dir, exist_ok=True)
                self.logger.debug(f"Ensuring directory exists: {save_dir}")
                
                # 测试目录写入权限
                test_file = os.path.join(save_dir, "test_write.tmp")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    self.logger.debug(f"Directory {save_dir} is writable")
                except Exception as e:
                    self.logger.error(f"Directory {save_dir} write test failed: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # 尝试创建一个不同的目录
                    save_dir = "./images"
                    os.makedirs(save_dir, exist_ok=True)
                    self.logger.debug(f"Using alternative directory: {save_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create directory: {str(e)}")
                self.logger.error(traceback.format_exc())
                save_dir = "."  # 使用当前目录作为备用
                self.logger.debug(f"Cannot create directory, using current directory as fallback")
            
            # 准备数据
            epochs = [m["epoch"] for m in self.metrics_history]
            train_loss = [m["train_loss"] for m in self.metrics_history]
            val_loss = [m["val_loss"] for m in self.metrics_history]
            train_auc = [m.get("train_auc", 0) for m in self.metrics_history]
            val_auc = [m.get("val_auc", 0) for m in self.metrics_history]
            train_auprc = [m.get("train_auprc", 0) for m in self.metrics_history]
            val_auprc = [m.get("val_auprc", 0) for m in self.metrics_history]
            train_f1 = [m.get("train_f1", 0) for m in self.metrics_history]
            val_f1 = [m.get("val_f1", 0) for m in self.metrics_history]
            
            # 获取测试集指标 (如果存在)
            test_metrics = getattr(self, 'test_metrics', None)
            
            self.logger.debug(f"Training curve data: Metrics for {len(epochs)} epochs")
            if test_metrics:
                self.logger.debug(f"Test set metrics: {test_metrics}")
            else:
                self.logger.info("Test set metrics not calculated yet (normal during training), the final chart will include test set results after training is complete.")
            
            # 1. 训练指标图表 (类似train.py的格式)
            try:
                self.logger.debug("Starting generation of training metrics plot...")
                
                import matplotlib
                matplotlib.use('Agg')
                
                plt.figure(figsize=(15, 10))
                final_epoch = epochs[-1] if epochs else 0
                
                # Loss Curve
                plt.subplot(2, 3, 1)
                plt.plot(epochs, train_loss, 'o-', color='blue', label='Training Loss')
                plt.plot(epochs, val_loss, 'o-', color='red', label='Validation Loss')
                if test_metrics:
                    plt.plot(final_epoch, test_metrics['test_loss'], '* ', markersize=10, color='purple', label=f"Test Loss ({test_metrics['test_loss']:.4f})")
                plt.title('Loss Curve')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # AUC Curve
                plt.subplot(2, 3, 2)
                plt.plot(epochs, train_auc, 'o-', color='dodgerblue', label='Training AUC')
                plt.plot(epochs, val_auc, 'o-', color='orange', label='Validation AUC')
                if test_metrics:
                    plt.plot(final_epoch, test_metrics['test_auc'], '* ', markersize=10, color='darkorange', label=f"Test AUC ({test_metrics['test_auc']:.4f})")
                plt.title('AUC Curve')
                plt.xlabel('Epochs')
                plt.ylabel('AUC')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # AUPRC Curve
                plt.subplot(2, 3, 3)
                plt.plot(epochs, train_auprc, 'o-', color='mediumseagreen', label='Training AUPRC')
                plt.plot(epochs, val_auprc, 'o-', color='green', label='Validation AUPRC')
                if test_metrics:
                    plt.plot(final_epoch, test_metrics['test_auprc'], '* ', markersize=10, color='darkgreen', label=f"Test AUPRC ({test_metrics['test_auprc']:.4f})")
                plt.title('AUPRC Curve')
                plt.xlabel('Epochs')
                plt.ylabel('AUPRC')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # F1 Score Curve
                plt.subplot(2, 3, 4)
                plt.plot(epochs, train_f1, 'o-', color='lightcoral', label='Training F1')
                plt.plot(epochs, val_f1, 'o-', color='red', label='Validation F1')
                if test_metrics:
                    plt.plot(final_epoch, test_metrics['test_f1'], '* ', markersize=10, color='darkred', label=f"Test F1 ({test_metrics['test_f1']:.4f})")
                plt.title('F1 Score Curve')
                plt.xlabel('Epochs')
                plt.ylabel('F1 Score')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # 保存训练指标图
                metrics_plot_path = os.path.join(save_dir, "training_metrics.png")
                self.logger.debug(f"Attempting to save training metrics plot to: {metrics_plot_path}")
                plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.plots["training_metrics"] = metrics_plot_path.replace("static/", "")
                self.logger.debug(f"Training metrics plot saved to {metrics_plot_path}")
                
                # 验证文件是否确实被保存
                if os.path.exists(metrics_plot_path):
                    file_size = os.path.getsize(metrics_plot_path)
                    self.logger.debug(f"Confirmed file saved: {metrics_plot_path} (Size: {file_size/1024:.1f}KB)")
                    
                    # 如果文件太小，可能是空图片
                    if file_size < 1000:
                        self.logger.warning(f"Warning: Training metrics plot file size is too small ({file_size} bytes)")
                else:
                    self.logger.error(f"File save failed: {metrics_plot_path} does not exist!")
                
            except Exception as e:
                self.logger.error(f"Error generating training metrics plot: {str(e)}")
                self.logger.error(traceback.format_exc())
                
            # 2. 混淆矩阵热力图（需要真实标签和预测结果）
            try:
                self.logger.debug("Starting generation of confusion matrix heatmap...")
                
                if hasattr(self, 'confusion_matrix') and self.confusion_matrix is not None:
                    self.logger.debug(f"Preparing to generate confusion matrix heatmap, confusion matrix:\n{self.confusion_matrix}")
                    
                    # 重置图形状态
                    plt.figure(figsize=(10, 7))
                    
                    # 检查混淆矩阵维度是否正确
                    if len(self.confusion_matrix.shape) != 2:
                        self.logger.error(f"Confusion matrix dimensions incorrect: {self.confusion_matrix.shape}, should be 2D")
                        # 创建一个默认的2x2混淆矩阵
                        self.confusion_matrix = np.array([[0, 0], [0, 0]])
                    
                    # 确保混淆矩阵是数值类型
                    cm = self.confusion_matrix.astype(np.int64)
                    
                    # 使用seaborn绘制热力图
                    try:
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title('Confusion Matrix Heatmap')
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        
                        # 保存混淆矩阵热力图
                        cm_plot_path = os.path.join(save_dir, "confusion_matrix_heatmap.png")
                        self.logger.debug(f"Attempting to save confusion matrix heatmap to: {cm_plot_path}")
                        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        self.plots["confusion_matrix"] = cm_plot_path.replace("static/", "")
                        self.logger.debug(f"Confusion matrix heatmap saved to {cm_plot_path}")
                        
                        # 验证文件是否确实被保存
                        if os.path.exists(cm_plot_path):
                            file_size = os.path.getsize(cm_plot_path)
                            self.logger.debug(f"Confirmed file saved: {cm_plot_path} (Size: {file_size/1024:.1f}KB)")
                            
                            # 如果文件太小，可能是空图片
                            if file_size < 1000:
                                self.logger.warning(f"Warning: Confusion matrix heatmap file size is too small ({file_size} bytes)")
                        else:
                            self.logger.error(f"File save failed: {cm_plot_path} does not exist!")
                            
                    except Exception as e:
                        self.logger.error(f"Error plotting and saving confusion matrix heatmap: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        
                        # 尝试更简单的方法生成热力图
                        try:
                            plt.figure(figsize=(8, 6))
                            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                            plt.title('Confusion Matrix')
                            plt.colorbar()
                            
                            # 添加文本标注
                            thresh = cm.max() / 2.
                            for i in range(cm.shape[0]):
                                for j in range(cm.shape[1]):
                                    plt.text(j, i, format(cm[i, j], 'd'),
                                             ha="center", va="center",
                                             color="white" if cm[i, j] > thresh else "black")
                            
                            plt.ylabel('True label')
                            plt.xlabel('Predicted label')
                            plt.tight_layout()
                            
                            # 使用备用文件名
                            cm_plot_path = os.path.join(save_dir, "confusion_matrix_simple.png")
                            plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            
                            self.logger.debug(f"Generated confusion matrix image using simple method: {cm_plot_path}")
                            self.plots["confusion_matrix"] = cm_plot_path.replace("static/", "")
                        except Exception as e2:
                            self.logger.error(f"Generating confusion matrix image using alternative method also failed: {str(e2)}")
                else:
                    self.logger.warning("Skipping confusion matrix heatmap generation as confusion matrix data is missing")
                    if not hasattr(self, 'confusion_matrix'):
                        self.logger.warning("Object does not have confusion_matrix attribute!")
                    elif self.confusion_matrix is None:
                        self.logger.warning("confusion_matrix attribute is None!")
                    
                    # 创建一个示例混淆矩阵，以便仍然能提供可视化
                    try:
                        self.logger.debug("Creating an example confusion matrix image...")
                        example_cm = np.array([[0, 0], [0, 0]])
                        plt.figure(figsize=(8, 6))
                        plt.imshow(example_cm, interpolation='nearest', cmap=plt.cm.Blues)
                        plt.title('Example Confusion Matrix (No Data)')
                        plt.colorbar()
                        plt.ylabel('True label')
                        plt.xlabel('Predicted label')
                        plt.tight_layout()
                        
                        cm_plot_path = os.path.join(save_dir, "confusion_matrix_example.png")
                        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        self.logger.debug(f"Created example confusion matrix image: {cm_plot_path}")
                        self.plots["confusion_matrix"] = cm_plot_path.replace("static/", "")
                    except Exception as e:
                        self.logger.error(f"Failed to create example confusion matrix image: {str(e)}")
                        
            except Exception as e:
                self.logger.error(f"Error generating confusion matrix heatmap: {str(e)}")
                self.logger.error(traceback.format_exc())
            
            # 更新进度
            await self._notify_progress()
            self.logger.info("All training plots and heatmap generation completed")
            
        except Exception as e:
            self.logger.error(f"Error generating training plots: {str(e)}")
            self.logger.error(traceback.format_exc())
        
    async def _save_model(self, dataset, best_threshold):
        """Save the trained model"""
        if self.best_model is None:
            self.logger.warning("Cannot save model, best_model is None")
            return
        
        self.logger.info(f"Starting model saving, dataset: {dataset}")
        
        try:
            # 确保目录存在
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存模型状态字典
            model_path = os.path.join(model_dir, f"model_{dataset}.h5")
            
            # 记录元数据
            metadata = {
                "dataset": dataset,
                "training_epochs": self.total_epochs,
                "best_threshold": float(best_threshold),
                "performance": {
                    "auc": float(self.current_metrics.get("val_auc", 0)),
                    "auprc": float(self.current_metrics.get("val_auprc", 0)),
                    "f1": float(self.current_metrics.get("val_f1", 0)),
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 设置为CPU张量再保存
            cpu_state_dict = {k: v.cpu() for k, v in self.best_model.state_dict().items()}
            
            # 一些检查
            total_params = sum(p.numel() for p in self.best_model.parameters())
            state_dict_size = sum(tensor.numel() for tensor in cpu_state_dict.values())
            self.logger.debug(f"Model statistics - Total parameters: {total_params:,}, State dict size: {state_dict_size:,}")
            
            # 使用h5py保存模型
            with h5py.File(model_path, 'w') as f:
                # 保存模型配置
                config_group = f.create_group('config')
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        subgroup = config_group.create_group(key)
                        for subkey, subvalue in value.items():
                            subgroup.attrs[subkey] = str(subvalue)
                    else:
                        config_group.attrs[key] = str(value)
                
                # 保存模型权重
                weights_group = f.create_group('weights')
                for name, param in cpu_state_dict.items():
                    weights_group.create_dataset(name, data=param.numpy())
            
            self.logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    async def _notify_progress(self, error=None, message_text=None):
        """Notify frontend about training progress"""
        try:
            message = {
                "type": "training_status",
                "status": self.status,
                "progress": self.progress,
                "epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "current_batch": self.current_batch,
                "total_batches": self.total_batches,
                "metrics": self.current_metrics
            }
            
            # 添加可选消息
            if message_text:
                message["message"] = message_text
                
            # 添加错误信息
            if error:
                message["error"] = str(error)
                
            # 添加已用时间
            if self.start_time:
                message["elapsed_time"] = time.time() - self.start_time
            
            # 发送到所有活动连接
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    self.logger.error(f"Error sending progress update: {str(e)}")
                    
            self.logger.debug(f"Sent progress update: Status={self.status}, Progress={self.progress}%, Epoch={self.current_epoch}/{self.total_epochs}")
            
        except Exception as e:
            self.logger.error(f"_notify_progress error: {str(e)}")
            self.logger.error(traceback.format_exc())