import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable
from torch.utils import data
import matplotlib.patches as patches
import logging
import traceback
import sys
from datetime import datetime
import time
from typing import Optional, Dict, Any, Tuple, List

from models import BIN_Interaction_Flat
from config import BIN_config_DBPE
from stream import BIN_Data_Encoder

# 配置日志系统
def setup_heatmap_logging():
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 获取热力图专用日志器
    logger = logging.getLogger("HeatmapGenerator")
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器，每天一个日志文件
        log_filename = f"logs/heatmap_{datetime.now().strftime('%Y%m%d')}.log"
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

class HeatmapGenerator:
    # 热力图生成类，用于药物-蛋白相互作用的可视化展示
    # Heatmap generation class for visualizing drug-protein interactions
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = None
        self.model = None
        self.logger = setup_heatmap_logging()
        # self.logger.info(f"热力图生成器初始化完成，使用设备: {self.device}")
        self.logger.info(f"HeatmapGenerator initialized, using device: {self.device}")

    def load_model(self, dataset, device=None):
        """Load the specified training model"""
        # self.logger.info(f"开始加载模型，数据集: {dataset}")
        self.logger.info(f"Starting model loading, dataset: {dataset}")
        start_time = time.time()
        
        try:
            if device:
                self.device = device
                # self.logger.info(f"使用指定设备: {device}")
                self.logger.info(f"Using specified device: {device}")
            
            # 获取模型配置
            self.config = BIN_config_DBPE()
            # self.logger.debug(f"模型配置: {self.config}")
            self.logger.debug(f"Model configuration: {self.config}")
            
            # 初始化模型
            self.model = BIN_Interaction_Flat(**self.config)
            # self.logger.info(f"创建模型实例: {self.model.__class__.__name__}")
            self.logger.info(f"Created model instance: {self.model.__class__.__name__}")
            
            # 加载模型权重
            model_path = f"saved_models/model_{dataset}.h5"
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                # error_msg = f"模型文件不存在: {model_path}"
                error_msg = f"Model file not found: {model_path}"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            # self.logger.info(f"从 {model_path} 加载模型权重")
            self.logger.info(f"Loading model weights from {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                # 新格式
                # self.logger.debug("检测到新格式的模型文件")
                self.logger.debug("Detected new format model file")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                # 读取元数据
                if 'metadata' in checkpoint:
                    metadata = checkpoint['metadata']
                    # self.logger.info(f"模型元数据: 数据集={metadata.get('dataset')}, 训练轮次={metadata.get('training_epochs')}, 最佳阈值={metadata.get('best_threshold')}")
                    self.logger.info(f"Model metadata: Dataset={metadata.get('dataset')}, Training Epochs={metadata.get('training_epochs')}, Best Threshold={metadata.get('best_threshold')}")
            else:
                # 旧格式
                # self.logger.debug("检测到旧格式的模型文件，尝试直接加载")
                self.logger.debug("Detected old format model file, attempting direct load")
                self.model.load_state_dict(checkpoint)
            
            # 将模型移至指定设备
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            # 计算模型参数
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # self.logger.info(f"模型总参数: {total_params:,}，可训练参数: {trainable_params:,}")
            self.logger.info(f"Model total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
            
            load_time = time.time() - start_time
            # self.logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
            self.logger.info(f"Model loading complete, time taken: {load_time:.2f} seconds")
            
            return {"status": "success"}
            
        except Exception as e:
            # self.logger.error(f"加载模型时出错: {str(e)}")
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

    def generate_heatmap(self, dataset, drug_smiles, target_seq, sample_idx, heatmap_type):
        """Generate heatmap based on given drug SMILES and protein sequence"""
        if self.model is None:
            # error_msg = "模型未加载，请先调用 load_model"
            error_msg = "Model not loaded, please call load_model first"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        # self.logger.info(f"开始生成热力图 - 类型: {heatmap_type}, 样本索引: {sample_idx}")
        # self.logger.debug(f"输入数据 - 药物SMILES长度: {len(drug_smiles)}, 蛋白质序列长度: {len(target_seq)}")
        self.logger.info(f"Starting heatmap generation - Type: {heatmap_type}, Sample index: {sample_idx}")
        self.logger.debug(f"Input data - Drug SMILES length: {len(drug_smiles)}, Protein sequence length: {len(target_seq)}")
        
        start_time = time.time()
        
        try:
            # 获取数据集路径
            dataset_path = get_task(dataset)
            # self.logger.debug(f"数据集路径: {dataset_path}")
            self.logger.debug(f"Dataset path: {dataset_path}")
            
            # 读取测试数据
            df_test = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
            # self.logger.info(f"测试集样本数: {len(df_test)}")
            self.logger.info(f"Test set sample count: {len(df_test)}")
            
            processing_start = time.time()
            
            # 数据处理逻辑
            dataFolder = get_task(dataset)
            encoding_start = time.time()
            
            # 创建数据编码器和加载器
            test_data = BIN_Data_Encoder([sample_idx], [0], df_test)
            # self.logger.debug(f"创建测试数据编码器，样本索引: {sample_idx}")
            self.logger.debug(f"Created test data encoder, sample index: {sample_idx}")
            
            # 获取样本数据
            d, p, d_mask, p_mask, _ = test_data[0]
            
            # 扩展维度
            d = torch.unsqueeze(d, 0).to(self.device)
            p = torch.unsqueeze(p, 0).to(self.device)
            d_mask = torch.unsqueeze(d_mask, 0).to(self.device)
            p_mask = torch.unsqueeze(p_mask, 0).to(self.device)
            
            encoding_time = time.time() - encoding_start
            # self.logger.debug(f"数据编码完成，耗时: {encoding_time:.2f}秒")
            self.logger.debug(f"Data encoding complete, time taken: {encoding_time:.2f} seconds")
            
            # 根据指定类型生成热力图
            if heatmap_type == 'attention':
                # self.logger.info("生成注意力热力图")
                self.logger.info("Generating attention heatmap")
                heatmap_data = self._get_attention_map(d, p, d_mask, p_mask)
            elif heatmap_type == 'interaction':
                # self.logger.info("生成相互作用强度热力图")
                self.logger.info("Generating interaction strength heatmap")
                heatmap_data = self._get_interaction_map(d, p, d_mask, p_mask)
            elif heatmap_type == 'gradient':
                # self.logger.info("生成梯度热力图")
                self.logger.info("Generating gradient heatmap")
                heatmap_data = self._get_gradient_map(d, p, d_mask, p_mask)
            else:
                # error_msg = f"不支持的热力图类型: {heatmap_type}"
                error_msg = f"Unsupported heatmap type: {heatmap_type}"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            processing_time = time.time() - processing_start
            # self.logger.debug(f"热力图数据处理完成，耗时: {processing_time:.2f}秒")
            self.logger.debug(f"Heatmap data processing complete, time taken: {processing_time:.2f} seconds")
            
            # 绘制热力图
            visualization_start = time.time()
            heatmap_path = self._plot_heatmap(drug_smiles, target_seq, d_mask, p_mask, heatmap_data, heatmap_type, sample_idx)
            visualization_time = time.time() - visualization_start
            # self.logger.debug(f"热力图可视化完成，耗时: {visualization_time:.2f}秒")
            self.logger.debug(f"Heatmap visualization complete, time taken: {visualization_time:.2f} seconds")
            
            total_time = time.time() - start_time
            # self.logger.info(f"热力图生成完成，总耗时: {total_time:.2f}秒，保存路径: {heatmap_path}")
            self.logger.info(f"Heatmap generation complete, total time: {total_time:.2f} seconds, save path: {heatmap_path}")
            
            return {
                "status": "success",
                "heatmap_path": heatmap_path,
                "processing_time": processing_time,
                "visualization_time": visualization_time,
                "total_time": total_time
            }
            
        except Exception as e:
            # self.logger.error(f"生成热力图时出错: {str(e)}")
            self.logger.error(f"Error generating heatmap: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

    def _get_attention_map(self, d, p, d_mask, p_mask):
        """Extract attention weights"""
        # self.logger.debug("提取注意力权重")
        self.logger.debug("Extracting attention weights")
        attention_start = time.time()
        
        try:
            # 获取注意力矩阵
            attentions = self.model.get_attention_map(d, p, d_mask, p_mask)
            
            # 移到CPU并转换为numpy数组
            attention_map = attentions[0].detach().cpu().numpy()  # 获取第一个样本的注意力图
            attention_map = attention_map[0]  # 获取第一个注意力头
            
            # self.logger.debug(f"注意力权重提取完成 - 形状: {attention_map.shape}")
            # self.logger.debug(f"注意力权重范围: [{np.min(attention_map):.4f}, {np.max(attention_map):.4f}]")
            self.logger.debug(f"Attention weight extraction complete - Shape: {attention_map.shape}")
            self.logger.debug(f"Attention weight range: [{np.min(attention_map):.4f}, {np.max(attention_map):.4f}]")
            
            attention_time = time.time() - attention_start
            # self.logger.debug(f"注意力权重计算耗时: {attention_time:.2f}秒")
            self.logger.debug(f"Attention weight calculation time: {attention_time:.2f} seconds")
            
            return attention_map
            
        except Exception as e:
            # self.logger.error(f"提取注意力权重时出错: {str(e)}")
            self.logger.error(f"Error extracting attention weights: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _get_interaction_map(self, d, p, d_mask, p_mask):
        """Extract interaction strength"""
        # self.logger.debug("提取相互作用强度")
        self.logger.debug("Extracting interaction strength")
        interaction_start = time.time()
        
        try:
            # 获取相互作用矩阵
            interaction_map = self.model.get_interaction_map(d, p, d_mask, p_mask)
            
            # 移到CPU并转换为numpy数组
            interaction_map = interaction_map[0].detach().cpu().numpy()  # 获取第一个样本的相互作用图
            
            # self.logger.debug(f"相互作用强度提取完成 - 形状: {interaction_map.shape}")
            # self.logger.debug(f"相互作用强度范围: [{np.min(interaction_map):.4f}, {np.max(interaction_map):.4f}]")
            self.logger.debug(f"Interaction strength extraction complete - Shape: {interaction_map.shape}")
            self.logger.debug(f"Interaction strength range: [{np.min(interaction_map):.4f}, {np.max(interaction_map):.4f}]")
            
            interaction_time = time.time() - interaction_start
            # self.logger.debug(f"相互作用强度计算耗时: {interaction_time:.2f}秒")
            self.logger.debug(f"Interaction strength calculation time: {interaction_time:.2f} seconds")
            
            return interaction_map
            
        except Exception as e:
            # self.logger.error(f"提取相互作用强度时出错: {str(e)}")
            self.logger.error(f"Error extracting interaction strength: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _get_gradient_map(self, d, p, d_mask, p_mask):
        """Extract gradient map"""
        # self.logger.debug("提取梯度权重")
        self.logger.debug("Extracting gradient weights")
        gradient_start = time.time()
        
        try:
            # 设置requires_grad
            d.requires_grad = True
            p.requires_grad = True
            
            # 前向传播
            output = self.model(d, p, d_mask, p_mask)
            
            # 创建目标向量 (for 梯度计算)
            target = torch.ones(output.size(), device=self.device)
            
            # 清除之前的梯度
            self.model.zero_grad()
            
            # 反向传播
            output.backward(target)
            
            # 提取药物和蛋白质嵌入的梯度
            d_grad = self.model.drug_embedding.weight.grad
            p_grad = self.model.protein_embedding.weight.grad
            
            # 计算梯度的绝对值
            d_grad_abs = torch.abs(d_grad).sum(dim=1)
            p_grad_abs = torch.abs(p_grad).sum(dim=1)
            
            # 维度转换
            d_values = d.view(-1)  # 药物输入
            p_values = p.view(-1)  # 蛋白质输入
            
            # 初始化梯度矩阵
            gradient_map = torch.zeros(d.size(1), p.size(1), device=self.device)
            
            # 填充梯度矩阵
            for i in range(d.size(1)):
                for j in range(p.size(1)):
                    if d_mask[0, i] == 0 or p_mask[0, j] == 0:
                        continue
                    d_idx = d_values[i]
                    p_idx = p_values[j]
                    gradient_map[i, j] = d_grad_abs[d_idx] * p_grad_abs[p_idx]
            
            # 移到CPU并转换为numpy数组
            gradient_map = gradient_map.detach().cpu().numpy()
            
            # self.logger.debug(f"梯度图提取完成 - 形状: {gradient_map.shape}")
            # self.logger.debug(f"梯度值范围: [{np.min(gradient_map):.4f}, {np.max(gradient_map):.4f}]")
            self.logger.debug(f"Gradient map extraction complete - Shape: {gradient_map.shape}")
            self.logger.debug(f"Gradient value range: [{np.min(gradient_map):.4f}, {np.max(gradient_map):.4f}]")
            
            gradient_time = time.time() - gradient_start
            # self.logger.debug(f"梯度图计算耗时: {gradient_time:.2f}秒")
            self.logger.debug(f"Gradient map calculation time: {gradient_time:.2f} seconds")
            
            return gradient_map
            
        except Exception as e:
            # self.logger.error(f"提取梯度图时出错: {str(e)}")
            self.logger.error(f"Error extracting gradient map: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _plot_heatmap(self, drug_smiles, target_seq, d_mask, p_mask, heatmap_data, heatmap_type, sample_idx):
        """Plot and save the heatmap"""
        # self.logger.debug(f"开始绘制热力图 - 类型: {heatmap_type}")
        self.logger.debug(f"Starting heatmap plotting - Type: {heatmap_type}")
        plot_start = time.time()
        
        try:
            # 创建保存目录
            save_dir = os.path.join('static', 'images', 'heatmaps')
            os.makedirs(save_dir, exist_ok=True)
            
            # 处理mask，找出有效的药物和蛋白质序列
            d_mask_np = d_mask[0].cpu().numpy()
            p_mask_np = p_mask[0].cpu().numpy()
            
            valid_d_indices = [i for i, m in enumerate(d_mask_np) if m == 1]
            valid_p_indices = [i for i, m in enumerate(p_mask_np) if m == 1]
            
            # self.logger.debug(f"有效药物序列长度: {len(valid_d_indices)}, 有效蛋白质序列长度: {len(valid_p_indices)}")
            self.logger.debug(f"Valid drug sequence length: {len(valid_d_indices)}, Valid protein sequence length: {len(valid_p_indices)}")
            
            # 截取有效部分
            valid_heatmap = heatmap_data[valid_d_indices, :][:, valid_p_indices]
            
            # 准备用于显示的序列
            drug_sequence = list(drug_smiles)
            protein_sequence = list(target_seq)
            
            valid_drug = [drug_sequence[i] for i in valid_d_indices]
            valid_protein = [protein_sequence[i] for i in valid_p_indices]
            
            # self.logger.debug(f"热力图有效区域形状: {valid_heatmap.shape}")
            self.logger.debug(f"Heatmap valid region shape: {valid_heatmap.shape}")
            
            # 确定热力图尺寸
            if len(valid_drug) > 50 or len(valid_protein) > 50:
                # self.logger.info(f"序列较长，调整图表大小 - 药物: {len(valid_drug)}, 蛋白质: {len(valid_protein)}")
                self.logger.info(f"Sequence is long, adjusting chart size - Drug: {len(valid_drug)}, Protein: {len(valid_protein)}")
                plt.figure(figsize=(max(10, len(valid_protein) * 0.2), max(8, len(valid_drug) * 0.2)))
            else:
                plt.figure(figsize=(10, 8))
            
            # 绘制热力图
            ax = sns.heatmap(
                valid_heatmap, 
                xticklabels=valid_protein, 
                yticklabels=valid_drug,
                cmap='viridis',
                cbar_kws={'label': f'{heatmap_type.capitalize()} Score'}
            )
            
            # 设置标题和轴标签
            plt.title(f"{heatmap_type.capitalize()} Heatmap")
            plt.xlabel("Protein Sequence")
            plt.ylabel("Drug SMILES")
            
            # 保存热力图
            heatmap_filename = f"{heatmap_type}_heatmap_{sample_idx}.png"
            save_path = os.path.join(save_dir, heatmap_filename)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            # 返回相对路径（用于网页显示）
            relative_path = os.path.join('images', 'heatmaps', heatmap_filename)
            
            plot_time = time.time() - plot_start
            # self.logger.debug(f"热力图绘制完成，耗时: {plot_time:.2f}秒, 保存到: {save_path}")
            self.logger.debug(f"Heatmap plotting complete, time taken: {plot_time:.2f} seconds, saved to: {save_path}")
            
            return relative_path
            
        except Exception as e:
            # self.logger.error(f"绘制热力图时出错: {str(e)}")
            self.logger.error(f"Error plotting heatmap: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def generate_all_heatmap_types(self, dataset, drug_smiles, target_seq, sample_idx):
        """Generate all types of heatmaps"""
        # self.logger.info(f"开始生成所有类型的热力图，样本索引: {sample_idx}")
        self.logger.info(f"Starting generation of all heatmap types, sample index: {sample_idx}")
        
        try:
            results = {}
            heatmap_types = ['attention', 'interaction', 'gradient']
            
            for heatmap_type in heatmap_types:
                # self.logger.info(f"生成 {heatmap_type} 类型热力图")
                self.logger.info(f"Generating {heatmap_type} type heatmap")
                result = self.generate_heatmap(dataset, drug_smiles, target_seq, sample_idx, heatmap_type)
                
                if result["status"] == "success":
                    results[heatmap_type] = result["heatmap_path"]
                    # self.logger.info(f"{heatmap_type} 热力图生成成功: {result['heatmap_path']}")
                    self.logger.info(f"{heatmap_type} heatmap generated successfully: {result['heatmap_path']}")
                else:
                    results[heatmap_type] = None
                    # self.logger.error(f"{heatmap_type} 热力图生成失败: {result['message']}")
                    self.logger.error(f"{heatmap_type} heatmap generation failed: {result['message']}")
            
            success_count = sum(1 for path in results.values() if path is not None)
            # self.logger.info(f"热力图生成完成，成功生成 {success_count}/{len(heatmap_types)} 张热力图")
            self.logger.info(f"Heatmap generation complete, successfully generated {success_count}/{len(heatmap_types)} heatmaps")
            
            return {
                "status": "success" if success_count > 0 else "error",
                "heatmaps": results,
                "success_count": success_count
            }
            
        except Exception as e:
            # self.logger.error(f"生成所有热力图时出错: {str(e)}")
            self.logger.error(f"Error generating all heatmaps: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}