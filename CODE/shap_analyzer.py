import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from torch.utils import data
from typing import List, Dict, Optional
import requests
import logging
import sys
from datetime import datetime
import traceback
import json

# 导入MolTrans模型和工具
from stream import BIN_Data_Encoder

# 配置日志系统
def setup_logging():
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 创建文件处理器，每天一个日志文件
    log_filename = f"logs/moltrans_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

def get_task(task_name):
    """Get dataset path"""
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'

class SHAPAnalyzer:
    def __init__(self, model, active_connections=None):
        self.model = model
        self.active_connections = active_connections if active_connections is not None else []
        self.status = "not_started"
        self.progress = 0
        self.results = {}
        self.error_info = {"message": "", "traceback": ""}
        self.logger = logging.getLogger("SHAPAnalyzer")
        self.logger.info("SHAPAnalyzer initialized successfully")
        self.image_paths = {
            "summary_plot": "",
            "bar_plot": "",
            "beeswarm_plot": "",
            "waterfall_plot": "",
            "force_plot": "",
            "global_importance": ""
        }
    
    async def analyze(self, num_samples=30, sample_ids=None):
        """Execute SHAP analysis"""
        try:
            self.logger.info(f"Starting SHAP analysis: num_samples={num_samples}, sample_ids={sample_ids}")
            self.status = "running"
            self.progress = 0
            self.error_info = {"message": "", "traceback": ""}
            
            # 记录系统信息
            import platform
            import shap
            
            # 记录模型信息
            self.logger.info(f"Model info: {self.model.__class__.__name__}")
            
            # 通知前端分析已开始
            await self._notify_progress("Starting SHAP analysis...", 10)
            
            # 准备数据
            self.logger.info("Preparing sample data...")
            await self._notify_progress("Preparing sample data...", 25)
            
            try:
                # 准备数据的详细日志
                self.logger.debug("Starting data preparation")
                samples = self._prepare_data(num_samples, sample_ids)
                self.logger.debug(f"Sample data preparation complete, shape: {samples.shape if hasattr(samples, 'shape') else 'Unknown'}")
                
                # 准备背景数据
                self.logger.info("Preparing background data...")
                await self._notify_progress("Preparing background data...", 40)
                background = self._prepare_background(samples)
                self.logger.debug(f"Background data preparation complete, shape: {background.shape if hasattr(background, 'shape') else 'Unknown'}")
                
                # 执行SHAP分析
                self.logger.info("Starting SHAP value calculation...")
                await self._notify_progress("Calculating SHAP values...", 60)
                shap_values = self._compute_shap_values(samples, background)
                self.logger.info("SHAP value calculation completed")
                self.logger.debug(f"SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'Unknown'}")
                
                # 生成可视化
                self.logger.info("Generating visualizations...")
                await self._notify_progress("Generating visualizations...", 80)
                save_dir = "static/images"  # 文件物理存储在static/images目录
                visualization_paths = self._generate_visualizations(shap_values, samples, save_dir)
                self.logger.info(f"Visualization generation complete, visualization_paths contains images: {list(visualization_paths.keys())}")
                
                # 确保image_paths已被正确设置
                if visualization_paths:
                    self.logger.info("Updating image_paths...")
                    # 从路径中移除'static/'前缀用于前端访问
                    self.image_paths = {
                        "summary_plot": visualization_paths.get('summary_plot', '').replace('\\', '/').replace('static/', ''),
                        "bar_plot": visualization_paths.get('bar_plot', '').replace('\\', '/').replace('static/', ''),
                        "beeswarm_plot": visualization_paths.get('beeswarm_plot', '').replace('\\', '/').replace('static/', ''),
                        "waterfall_plot": visualization_paths.get('waterfall_plot', '').replace('\\', '/').replace('static/', ''),
                        "force_plot": visualization_paths.get('force_plot', '').replace('\\', '/').replace('static/', ''),
                        "global_importance": visualization_paths.get('global_importance', '').replace('\\', '/').replace('static/', '')
                    }
                    
                    # 如果global_importance为空但bar_plot存在，使用bar_plot代替
                    if not self.image_paths["global_importance"] and self.image_paths["bar_plot"]:
                        self.image_paths["global_importance"] = self.image_paths["bar_plot"]
                        self.logger.info("Using bar_plot path for global_importance")
                    
                    # 确保force_plot有值
                    if not self.image_paths["force_plot"] and visualization_paths.get('first_sample_force'):
                        self.image_paths["force_plot"] = visualization_paths.get('first_sample_force', '').replace('\\', '/').replace('static/', '')
                        self.logger.info("Using first_sample_force path for force_plot")
                    
                    # 添加其他样本图像路径
                    for key, path in visualization_paths.items():
                        if key not in self.image_paths:
                            self.image_paths[key] = path.replace('\\', '/').replace('static/', '')
                            self.logger.info(f"Adding extra image path: {key} -> {self.image_paths[key]}")
                    
                    self.logger.info(f"image_paths updated: {self.image_paths}")
                else:
                    self.logger.warning("visualization_paths is empty, cannot update image_paths")
                    # 确保使用正斜杠，同时移除static前缀
                    save_dir_normalized = save_dir.replace('\\', '/').replace('static/', '')
                    self.image_paths = {
                        "summary_plot": os.path.join(save_dir_normalized, 'shap_summary_plot.png').replace('\\', '/'),
                        "bar_plot": os.path.join(save_dir_normalized, 'shap_bar_plot.png').replace('\\', '/'),
                        "beeswarm_plot": os.path.join(save_dir_normalized, 'shap_beeswarm_plot.png').replace('\\', '/'),
                        "waterfall_plot": os.path.join(save_dir_normalized, 'shap_waterfall_plot.png').replace('\\', '/'),
                        "force_plot": os.path.join(save_dir_normalized, 'shap_force_plot.png').replace('\\', '/'),
                        "global_importance": os.path.join(save_dir_normalized, 'shap_bar_plot.png').replace('\\', '/')
                    }
                    self.logger.info(f"Setting image_paths using default paths: {self.image_paths}")
                
                # 计算统计信息
                self.logger.info("Calculating statistics...")
                await self._notify_progress("Calculating statistics...", 90)
                self._calculate_statistics(shap_values)
                self.logger.info("Statistics calculation complete")
                
                # 完成
                self.status = "completed"
                self.progress = 100
                self.logger.info("SHAP analysis completed successfully")
                self.logger.info(f"Final image_paths: {self.image_paths}")
                self.logger.info(f"Final results contain image paths: {self.results['images']}")
                await self._notify_progress("SHAP analysis completed", 100)
                
            except Exception as e:
                # 详细捕获错误
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                self.status = "error"
                self.progress = 0
                self.error_info = {
                    "message": error_msg,
                    "traceback": error_traceback,
                    "stage": self.progress,
                    "timestamp": time.time()
                }
                self.logger.error(f"SHAP analysis error: {error_msg}")
                self.logger.error(f"Detailed error information:\n{error_traceback}")
                # Add logging to check for _notify_error before calling
                self.logger.debug(f"Checking existence of _notify_error: {hasattr(self, '_notify_error')}")
                if hasattr(self, '_notify_error'):
                    await self._notify_error(error_msg, error_traceback)
                else:
                    self.logger.error("_notify_error method not found on self!")
                
        except Exception as outer_e:
            # 捕获分析过程中的任何错误
            self.status = "error"
            self.progress = 0
            self.error_info = {
                "message": str(outer_e),
                "traceback": traceback.format_exc(),
                "stage": "initialization",
                "timestamp": time.time()
            }
            self.logger.error(f"SHAP analysis initialization error: {str(outer_e)}")
            self.logger.error(f"Detailed error information:\n{traceback.format_exc()}")
    
    def _prepare_data(self, num_samples, sample_ids=None):
        """Prepare sample data for SHAP analysis"""
        self.logger.debug(f"_prepare_data started: num_samples={num_samples}, sample_ids={sample_ids}")
        
        # 获取数据集
        try:
            # 加载测试数据集
            dataFolder = get_task('biosnap')  # 默认使用biosnap
            df_test = pd.read_csv(os.path.join(dataFolder, 'test.csv'))
            
            # 创建数据加载器
            small_batch_size = 2
            test_generator = data.DataLoader(
                BIN_Data_Encoder(df_test.index, df_test.Label, df_test),
                batch_size=small_batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=False
            )
            
            # 收集样本数据
            all_d = []
            all_p = []
            all_d_mask = []
            all_p_mask = []
            self.labels = []
            
            # 使用指定的设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 提取样本数据
            with torch.no_grad():
                for i, (d, p, d_mask, p_mask, label) in enumerate(test_generator):
                    if len(all_d) >= num_samples: # Check collected samples, not batch index
                        break
                        
                    # 添加每个批次中的样本，直到达到 num_samples
                    for j in range(len(d)):
                        if len(all_d) < num_samples:
                            all_d.append(d[j].to(device))
                            all_p.append(p[j].to(device))
                            all_d_mask.append(d_mask[j].to(device))
                            all_p_mask.append(p_mask[j].to(device))
                            self.labels.append(label[j].item())
                        else:
                            break # Reached num_samples within the batch
            
            # 获取序列长度
            self.drug_seq_len = len(all_d[0])
            self.protein_seq_len = len(all_p[0])
            
            # 计算总特征数
            self.num_features = self.drug_seq_len + self.protein_seq_len
            
            self.logger.debug(f"Drug sequence length: {self.drug_seq_len}, Protein sequence length: {self.protein_seq_len}")
            self.logger.debug(f"Total number of features: {self.num_features}")
            self.logger.debug(f"Collected {len(all_d)} samples for analysis")
            
            # 创建特征名称
            self.feature_names = []
            # 药物特征
            for i in range(self.drug_seq_len):
                self.feature_names.append(f"Drug_Pos_{i}")
            # 蛋白质特征
            for i in range(self.protein_seq_len):
                self.feature_names.append(f"Protein_Pos_{i}")
            
            # 存储收集的数据以供后续使用
            self.all_d = all_d
            self.all_p = all_p
            self.all_d_mask = all_d_mask
            self.all_p_mask = all_p_mask
            
            # 确定分析样本数和背景样本数
            self.num_analyze = min(num_samples, len(all_d)) # Use actual number of collected samples
            self.background_size = min(5, len(all_d)) # Background based on collected samples
            
            # 创建实例数据 (use self.num_analyze)
            instances_data = np.zeros((self.num_analyze, self.num_features))
            
            for i in range(self.num_analyze):
                idx = i # Use direct index from collected samples for analysis
                instances_data[i, :self.drug_seq_len] = all_d[idx].cpu().numpy()
                instances_data[i, self.drug_seq_len:] = all_p[idx].cpu().numpy()
            
            # 计算原始预测值，用于结果展示
            self.predictions = []
            with torch.no_grad():
                for i in range(self.num_analyze):
                    idx = i # Use direct index
                    output = self.model(all_d[idx].unsqueeze(0), all_p[idx].unsqueeze(0),
                                      all_d_mask[idx].unsqueeze(0), all_p_mask[idx].unsqueeze(0))
                    sigmoid_output = torch.sigmoid(output)
                    if sigmoid_output.numel() == 1:
                        self.predictions.append(sigmoid_output.item())
                    else:
                        self.predictions.append(sigmoid_output.mean().item())
            
            self.logger.debug("_prepare_data completed")
            return instances_data
            
        except Exception as e:
            self.logger.error(f"Error preparing sample data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise e
    
    def _prepare_background(self, samples):
        """Prepare background data for SHAP analysis"""
        self.logger.debug("_prepare_background started")
        
        try:
            # 创建背景数据 (use self.background_size)
            background_data = np.zeros((self.background_size, self.num_features))
            
            for i in range(self.background_size):
                # 合并药物和蛋白质数据为一个特征向量
                background_data[i, :self.drug_seq_len] = self.all_d[i].cpu().numpy()
                background_data[i, self.drug_seq_len:] = self.all_p[i].cpu().numpy()
            
            self.background_data = background_data
            self.logger.debug(f"Background data prepared, shape: {background_data.shape}")
            return background_data
            
        except Exception as e:
            self.logger.error(f"Error preparing background data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise e
    
    def _compute_shap_values(self, samples, background):
        """Compute SHAP values"""
        self.logger.debug("_compute_shap_values started")
        
        try:
            # 使用指定的设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 定义模型包装器函数
            def model_wrapper(masker_x):
                """
                Wrapper function for the model, ensures it returns results of the same length as the input.
                masker_x: Input data, could be background data or its perturbed versions.
                """
                with torch.no_grad():
                    # 获取当前输入批次大小
                    current_batch_size = masker_x.shape[0]
                    
                    # 确保结果数组与输入批次大小匹配
                    results = np.zeros(current_batch_size)
                    
                    # 处理每个输入样本
                    for i in range(current_batch_size):
                        x_single = masker_x[i:i+1]  # 取单个样本，保持2D形状
                        
                        # 将输入从NumPy数组转换为PyTorch张量
                        x_tensor = torch.tensor(x_single, dtype=torch.float32).to(device)
                        
                        # 分离药物和蛋白质数据
                        input_d = x_tensor[:, :self.drug_seq_len].long()
                        input_p = x_tensor[:, self.drug_seq_len:].long()
                        
                        # 创建掩码 (全1)
                        input_d_mask = torch.ones((1, self.drug_seq_len), device=device).long()
                        input_p_mask = torch.ones((1, self.protein_seq_len), device=device).long()
                        
                        # 运行模型
                        output = self.model(input_d, input_p, input_d_mask, input_p_mask)
                        
                        # 应用sigmoid并确保是标量
                        sigmoid_output = torch.sigmoid(output)
                        
                        # 提取标量值
                        if sigmoid_output.numel() == 1:
                            results[i] = sigmoid_output.item()
                        else:
                            results[i] = sigmoid_output.mean().item()
                    return results
            
            # 先测试模型包装器，确保输出形状正确
            self.logger.debug("Validating model wrapper output...")
            test_out = model_wrapper(background)
            self.logger.debug(f"Background data shape: {background.shape}")
            self.logger.debug(f"Model wrapper output shape: {test_out.shape if hasattr(test_out, 'shape') else 'Scalar'}")
            
            # 确保输出与背景数据长度匹配
            if len(test_out) != len(background):
                raise ValueError(f"Model wrapper output length {len(test_out)} does not match background data length {len(background)}!")
            
            self.logger.debug("Calculating SHAP values using KernelExplainer...")
            # 尝试使用KernelExplainer
            try:
                # 创建KernelExplainer并使用验证过的模型包装器
                self.explainer = shap.KernelExplainer(
                    model_wrapper, 
                    background,
                    link='identity'
                )
                
                # 使用较大的nsamples值提高精度
                self.logger.debug("Calculating sample SHAP values...")
                shap_values = self.explainer.shap_values(samples, nsamples=200)
                
                # 确保SHAP值是numpy数组
                if not isinstance(shap_values, np.ndarray):
                    shap_values = np.array(shap_values)
                
                # 确保SHAP值具有正确的维度 [样本数, 特征数]
                if len(shap_values.shape) > 2:
                    shap_values = shap_values.reshape(shap_values.shape[0], -1)
                
                self.logger.debug(f"KernelExplainer successful, SHAP values shape: {shap_values.shape}")
                    
            except Exception as e:
                self.logger.error(f"KernelExplainer error: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                # 尝试使用DeepExplainer替代方案
                self.logger.debug("Attempting DeepExplainer alternative...")
                
                try:
                    # 创建特殊的背景和实例张量
                    background_tensor_d = torch.stack([d for d in self.all_d[:self.background_size]])
                    background_tensor_p = torch.stack([p for p in self.all_p[:self.background_size]])
                    
                    instance_tensor_d = torch.stack([self.all_d[min(i + self.background_size, len(self.all_d) - 1)] 
                                                   for i in range(self.num_analyze)])
                    instance_tensor_p = torch.stack([self.all_p[min(i + self.background_size, len(self.all_p) - 1)] 
                                                   for i in range(self.num_analyze)])
                    
                    # 定义特殊包装模型函数
                    def wrapped_model(d, p):
                        return torch.sigmoid(self.model(
                            d, p, 
                            torch.ones((d.shape[0], self.drug_seq_len), device=device).long(),
                            torch.ones((p.shape[0], self.protein_seq_len), device=device).long()
                        ))
                    
                    # 创建DeepExplainer
                    deep_explainer = shap.DeepExplainer(wrapped_model, 
                                                       [background_tensor_d, background_tensor_p])
            
            # 计算SHAP值
                    deep_shap_values = deep_explainer.shap_values([instance_tensor_d, instance_tensor_p])
                    
                    # 处理药物和蛋白质的SHAP值
                    drug_shap = deep_shap_values[0] if isinstance(deep_shap_values, list) else deep_shap_values
                    protein_shap = deep_shap_values[1] if isinstance(deep_shap_values, list) else np.zeros_like(deep_shap_values)
                    
                    # 合并SHAP值为一个统一数组
                    combined_shap = []
                    for i in range(len(drug_shap)):
                        combined_shap.append(np.concatenate([drug_shap[i].flatten(), protein_shap[i].flatten()]))
                    
                    shap_values = np.array(combined_shap)
                    self.explainer = deep_explainer
                    self.logger.debug(f"DeepExplainer successful, SHAP values shape: {shap_values.shape}")
                    
                except Exception as e2:
                    self.logger.error(f"DeepExplainer also failed: {str(e2)}")
                    self.logger.error(traceback.format_exc())
                    
                    # 回退到手动计算特征重要性
                    self.logger.debug("Falling back to manual feature importance calculation...")
                    
                    # 创建简单的特征重要性矩阵
                    shap_values = np.zeros((self.num_analyze, self.num_features))
                    
                    # 对于每个要分析的样本
                    for i in range(self.num_analyze):
                        # 获取基准预测
                        baseline_pred = self.predictions[i]
                        
                        # 对于每个特征，计算简单的扰动重要性
                        for j in range(self.num_features):
                            # 创建扰动副本
                            perturbed_instance = samples[i].copy()
                            
                            # 扰动特征 (设置为0或均值)
                            perturbed_instance[j] = 0  # 或使用背景数据的均值
                            
                            # 计算扰动后的预测
                            perturbed_pred = model_wrapper(perturbed_instance.reshape(1, -1))[0]
                            
                            # 计算特征重要性 (预测差异)
                            shap_values[i, j] = baseline_pred - perturbed_pred
                    
                    # 创建简单的Explainer对象，只包含expected_value
                    class SimpleExplainer:
                        def __init__(self, expected_value):
                            self.expected_value = expected_value
                    
                    # 使用平均预测作为基准值
                    self.explainer = SimpleExplainer(np.mean(self.predictions))
                    self.logger.debug("Manual calculation successful, using prediction difference as SHAP values")
            
            # 存储SHAP值和分析用的样本
            self.shap_values = shap_values
            self.analyzed_samples = samples
            
            self.logger.debug(f"SHAP value calculation complete, value range: [{np.min(shap_values):.4f}, {np.max(shap_values):.4f}]")
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error computing SHAP values: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise e
    
    def _generate_visualizations(self, shap_values, samples, save_dir, top_k=20):
        """Generate SHAP visualizations using native SHAP library functions"""
        self.logger.debug("_generate_visualizations started")
        
        try:
            # 确保保存目录存在并标准化路径
            save_dir = save_dir.replace('\\', '/')
            
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)
            
            # 获取feature_names，如果不可用则创建通用名称
            if hasattr(self, 'feature_names') and self.feature_names is not None:
                feature_names = self.feature_names
            else:
                # 生成默认特征名称
                drug_features = [f"Drug_{i+1}" for i in range(self.drug_seq_len)]
                protein_features = [f"Protein_{i+1}" for i in range(self.protein_seq_len)]
                feature_names = drug_features + protein_features
            
            # 收集要返回的所有图像路径 - 使用物理保存路径
            visualization_paths = {}
            
            # 准备基准值（expected value）
            base_value = getattr(self.explainer, 'expected_value', 0)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0] if len(base_value) > 0 else 0
                
            # 创建Explanation对象以便使用新版SHAP函数
            explanation = shap.Explanation(
                values=shap_values,
                base_values=np.full(shap_values.shape[0], base_value),
                data=samples,
                feature_names=feature_names
            )
            
            # 1. 生成条形图 (Bar Plot)
            self._generate_bar_plot(explanation, shap_values, feature_names, save_dir, visualization_paths, top_k)
            
            # 2. 生成摘要图 (Summary Plot)
            self._generate_summary_plot(shap_values, samples, feature_names, save_dir, visualization_paths, top_k)
            
            # 3. 生成蜂群图 (Beeswarm Plot)
            self._generate_beeswarm_plot(explanation, shap_values, feature_names, save_dir, visualization_paths, top_k)
            
            # 4. 生成瀑布图 (Waterfall Plot)
            if shap_values.shape[0] > 0:
                self._generate_waterfall_plot(explanation, shap_values, samples, feature_names, base_value, save_dir, visualization_paths)
            
            # 5. 生成力图 (Force Plot)
            if shap_values.shape[0] > 0:
                self._generate_force_plot(explanation, shap_values, samples, feature_names, base_value, save_dir, visualization_paths)
            
            # 保存数据统计报告
            self.logger.debug("Saving visualization statistics report...")
            self._save_visualization_stats(shap_values, feature_names, save_dir, visualization_paths, top_k)
            
            self.logger.debug(f"Visualization generation complete, {len(visualization_paths)} images saved to {save_dir}")
            return visualization_paths
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # 返回空字典，表示没有成功生成可视化
            return {}
    
    def _generate_bar_plot(self, explanation, shap_values, feature_names, save_dir, visualization_paths, top_k=20):
        """Generate Bar Plot - Global Feature Importance"""
        self.logger.debug("Generating global feature importance bar plot (Bar Plot)...")
        
        # 安全地获取原始格式，如果不存在则为None
        original_float_format = plt.rcParams.get('axes.formatter.float_format')
        
        try:
            # 尝试设置新格式
            try:
                plt.rcParams['axes.formatter.float_format'] = '{:.4f}'.format
                self.logger.debug("Temporarily set matplotlib float format to {:.4f}")
            except KeyError:
                self.logger.warning("Could not set axes.formatter.float_format, version might not support it")

            
            plt.figure(figsize=(12, 8))
            shap.plots.bar(explanation, max_display=20, show=False)
            plt.tight_layout()
            bar_plot_path = os.path.join(save_dir, 'shap_bar_plot.png').replace('\\', '/')
            plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['bar_plot'] = bar_plot_path
            visualization_paths['global_importance'] = bar_plot_path
            self.logger.debug(f"Bar Plot saved to: {bar_plot_path}")
        except Exception as e:
            self.logger.error(f"Error generating Bar Plot: {str(e)}")
            # 备选方案：使用自定义实现
            plt.figure(figsize=(12, 8))
            mean_imp = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_imp)[-top_k:][::-1]
            plt.barh(
                [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in top_indices],
                mean_imp[top_indices]
            )
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'Top {top_k} features by importance')
            plt.tight_layout()
            bar_plot_path = os.path.join(save_dir, 'shap_bar_plot.png').replace('\\', '/')
            plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['bar_plot'] = bar_plot_path
            visualization_paths['global_importance'] = bar_plot_path
        finally:
            if original_float_format is not None:
                try:
                    plt.rcParams['axes.formatter.float_format'] = original_float_format
                    self.logger.debug("Restored original matplotlib float format")
                except KeyError:
                     self.logger.warning("Error restoring original format, might have been removed")
            else:
                plt.rcParams.pop('axes.formatter.float_format', None)
                self.logger.debug("Original format did not exist, attempting to remove potentially added format")
    
    def _generate_summary_plot(self, shap_values, samples, feature_names, save_dir, visualization_paths, top_k=20):
        """Generate Summary Plot"""
        self.logger.debug("Generating SHAP value summary plot (Summary Plot)...")
        
        original_float_format = plt.rcParams.get('axes.formatter.float_format')
        
        try:
            try:
                plt.rcParams['axes.formatter.float_format'] = '{:.4f}'.format
                self.logger.debug("Temporarily set matplotlib float format to {:.4f}")
            except KeyError:
                self.logger.warning("Could not set axes.formatter.float_format, version might not support it")

            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values, 
                samples,
                feature_names=feature_names,
                max_display=30,
                show=False,
                plot_size=(12, 10)
            )
            plt.tight_layout()
            summary_plot_path = os.path.join(save_dir, 'shap_summary_plot.png').replace('\\', '/')
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['summary_plot'] = summary_plot_path
            self.logger.debug(f"Summary Plot saved to: {summary_plot_path}")
        except Exception as e:
            self.logger.error(f"Error generating Summary Plot: {str(e)}")
            
            # 备选方案：使用自定义实现
            try:
                # 计算全局平均SHAP值绝对值
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # 创建摘要图
                plt.figure(figsize=(12, 10))
                
                # 对特征进行重新排序，按平均SHAP值大小
                sorted_indices = np.argsort(mean_abs_shap)
                sorted_feature_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in sorted_indices[-top_k:]]
                
                # 绘制摘要图
                for i, idx in enumerate(sorted_indices[-top_k:]):
                    # 获取此特征的所有样本的SHAP值
                    feature_shap_values = shap_values[:, idx]
                    
                    # 计算SHAP值的颜色
                    if samples is not None:
                        feature_values = samples[:, idx]
                        # 归一化特征值到[0,1]区间用于颜色映射
                        norm_values = (feature_values - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values) + 1e-10)
                    else:
                        # 使用SHAP值的符号作为颜色
                        norm_values = (feature_shap_values > 0).astype(float)
                    
                    # 创建颜色映射
                    colors = plt.cm.coolwarm(norm_values)
                    
                    # 在y位置i处绘制此特征的所有SHAP值
                    plt.scatter(
                        feature_shap_values,  # x值：SHAP值
                        np.ones(len(feature_shap_values)) * i,  # y值：固定在位置i
                        c=colors,  # 基于特征值的颜色
                        alpha=0.6,  # 透明度
                        s=20  # 点大小
                    )
                
                # 设置y轴刻度为特征名称
                plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
                plt.xlabel('SHAP values (impact magnitude and direction)')
                plt.title('SHAP Summary Plot (color represents feature value)')
                plt.axvline(x=0, color='silver', linestyle='-', alpha=0.5)  # 在x=0处添加垂直线
                plt.tight_layout()
                
                # 保存摘要图
                summary_plot_path = os.path.join(save_dir, 'shap_summary_plot.png').replace('\\', '/')
                plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths['summary_plot'] = summary_plot_path
                self.logger.debug(f"Fallback Summary Plot saved to: {summary_plot_path}")
            except Exception as backup_error:
                self.logger.error(f"Fallback summary plot generation also failed: {str(backup_error)}")
        finally:
            if original_float_format is not None:
                try:
                    plt.rcParams['axes.formatter.float_format'] = original_float_format
                    self.logger.debug("Restored original matplotlib float format")
                except KeyError:
                     self.logger.warning("Error restoring original format, might have been removed")
            else:
                plt.rcParams.pop('axes.formatter.float_format', None)
                self.logger.debug("Original format did not exist, attempting to remove potentially added format")
    
    def _generate_beeswarm_plot(self, explanation, shap_values, feature_names, save_dir, visualization_paths, top_k=20):
        """Generate Beeswarm Plot"""
        self.logger.debug("Generating SHAP value beeswarm plot (Beeswarm Plot)...")
        
        original_float_format = plt.rcParams.get('axes.formatter.float_format')
        
        try:
            try:
                plt.rcParams['axes.formatter.float_format'] = '{:.4f}'.format
                self.logger.debug("Temporarily set matplotlib float format to {:.4f}")
            except KeyError:
                self.logger.warning("Could not set axes.formatter.float_format, version might not support it")

            plt.figure(figsize=(14, 10))
            shap.plots.beeswarm(explanation, max_display=20, show=False)
            plt.tight_layout()
            beeswarm_plot_path = os.path.join(save_dir, 'shap_beeswarm_plot.png').replace('\\', '/')
            plt.savefig(beeswarm_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['beeswarm_plot'] = beeswarm_plot_path
            self.logger.debug(f"Beeswarm Plot saved to: {beeswarm_plot_path}")
        except Exception as e:
            self.logger.error(f"Error generating Beeswarm Plot: {str(e)}")
            
            # 备选方案：使用自定义实现
            try:
                # 计算全局平均SHAP值绝对值
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # 对特征进行重新排序，按平均SHAP值大小
                sorted_indices = np.argsort(mean_abs_shap)
                sorted_feature_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in sorted_indices[-top_k:]]
                
                plt.figure(figsize=(14, 10))
                
                # 蜂群图逻辑
                for i, idx in enumerate(sorted_indices[-top_k:]):
                    feature_shap_values = shap_values[:, idx]
                    # 使用抖动效果创建蜂群效果
                    y_pos = np.ones(len(feature_shap_values)) * i + np.random.normal(0, 0.1, len(feature_shap_values))
                    plt.scatter(
                        feature_shap_values,
                        y_pos,
                        c=plt.cm.coolwarm(0.5 + 0.5 * np.sign(feature_shap_values)),
                        alpha=0.6,
                        s=20
                    )
                
                plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
                plt.xlabel('SHAP values (impact magnitude and direction)')
                plt.title('SHAP Value Distribution (Beeswarm Plot)')
                plt.axvline(x=0, color='silver', linestyle='-', alpha=0.5)
                plt.tight_layout()
                
                beeswarm_plot_path = os.path.join(save_dir, 'shap_beeswarm_plot.png').replace('\\', '/')
                plt.savefig(beeswarm_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths['beeswarm_plot'] = beeswarm_plot_path
                self.logger.debug(f"Fallback Beeswarm Plot saved to: {beeswarm_plot_path}")
            except Exception as backup_error:
                self.logger.error(f"Fallback beeswarm plot generation also failed: {str(backup_error)}")
        finally:
            if original_float_format is not None:
                try:
                    plt.rcParams['axes.formatter.float_format'] = original_float_format
                    self.logger.debug("Restored original matplotlib float format")
                except KeyError:
                     self.logger.warning("Error restoring original format, might have been removed")
            else:
                plt.rcParams.pop('axes.formatter.float_format', None)
                self.logger.debug("Original format did not exist, attempting to remove potentially added format")
    
    def _generate_waterfall_plot(self, explanation, shap_values, samples, feature_names, base_value, save_dir, visualization_paths):
        """Generate Waterfall Plot"""
        self.logger.debug("Generating SHAP value waterfall plot (Waterfall Plot)...")
        
        original_float_format = plt.rcParams.get('axes.formatter.float_format')
        
        try:
            try:
                # 瀑布图通常需要更高的精度
                plt.rcParams['axes.formatter.float_format'] = '{:.5f}'.format 
                self.logger.debug("Temporarily set matplotlib float format to {:.5f}")
            except KeyError:
                self.logger.warning("Could not set axes.formatter.float_format, version might not support it")

            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(explanation[0], max_display=15, show=False)
            plt.tight_layout()
            waterfall_plot_path = os.path.join(save_dir, 'shap_waterfall_plot.png').replace('\\', '/')
            plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['waterfall_plot'] = waterfall_plot_path
            self.logger.debug(f"Waterfall Plot saved to: {waterfall_plot_path}")
        except Exception as e:
            self.logger.error(f"Error generating Waterfall Plot: {str(e)}")
            
            # 备选方案：使用自定义实现
            try:
                # 选择第一个样本
                sample_idx = 0
                sample_shap = shap_values[sample_idx]
                
                # 选择最重要的15个特征
                waterfall_feature_count = min(15, sample_shap.shape[0])
                sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:waterfall_feature_count]
                sorted_values = sample_shap[sorted_idx]
                sorted_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in sorted_idx]
                
                plt.figure(figsize=(12, 8))
                
                # 基准值条
                base_pred = base_value
                
                # 瀑布图组件
                cumulative = base_pred
                plot_data = []
                
                # 添加基准值
                plot_data.append(('Base value', base_pred, base_pred, 'gray'))
                
                # 添加每个特征的贡献
                for i, name in enumerate(sorted_names):
                    feature_impact = sorted_values[i]
                    old_cumulative = cumulative
                    cumulative += feature_impact
                    color = 'red' if feature_impact > 0 else 'blue'
                    plot_data.append((name, old_cumulative, cumulative, color))
                
                # 添加最终预测值
                plot_data.append(('Prediction', cumulative, cumulative, 'green'))
                
                # 绘制瀑布图
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 绘制基准条和最终预测条
                for i, (name, start, end, color) in enumerate([plot_data[0], plot_data[-1]]):
                    ax.barh(i, end, color=color, alpha=0.7)
                    ax.text(end/2, i, f"{name}: {end:.3f}", ha='center', va='center')
                
                # 绘制中间的贡献条
                for i, (name, start, end, color) in enumerate(plot_data[1:-1], 1):
                    ax.barh(i, end - start, left=start, color=color, alpha=0.7)
                    contribution = end - start
                    direction = '+' if contribution > 0 else ''
                    ax.text(max(start, end) + 0.05, i, f"{name}: {direction}{contribution:.3f}", va='center')
                
                ax.set_yticks(range(len(plot_data)))
                ax.set_yticklabels([p[0] for p in plot_data])
                ax.set_xlabel('Prediction Contribution')
                ax.set_title('SHAP Waterfall Plot - Sample Feature Contribution')
                
                plt.tight_layout()
                waterfall_plot_path = os.path.join(save_dir, 'shap_waterfall_plot.png').replace('\\', '/')
                plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths['waterfall_plot'] = waterfall_plot_path
                self.logger.debug(f"Fallback Waterfall Plot saved to: {waterfall_plot_path}")
            except Exception as backup_error:
                self.logger.error(f"Fallback waterfall plot generation also failed: {str(backup_error)}")
        finally:
            if original_float_format is not None:
                try:
                    plt.rcParams['axes.formatter.float_format'] = original_float_format
                    self.logger.debug("Restored original matplotlib float format")
                except KeyError:
                     self.logger.warning("Error restoring original format, might have been removed")
            else:
                plt.rcParams.pop('axes.formatter.float_format', None)
                self.logger.debug("Original format did not exist, attempting to remove potentially added format")
    
    def _generate_force_plot(self, explanation, shap_values, samples, feature_names, base_value, save_dir, visualization_paths):
        """Generate Force Plot"""
        self.logger.debug("Generating SHAP value force plot (Force Plot)...")
        
        original_float_format = plt.rcParams.get('axes.formatter.float_format')
        
        try:
            try:
                plt.rcParams['axes.formatter.float_format'] = '{:.4f}'.format
                self.logger.debug("Temporarily set matplotlib float format to {:.4f}")
            except KeyError:
                self.logger.warning("Could not set axes.formatter.float_format, version might not support it")

            force_plot_path = os.path.join(save_dir, 'shap_force_plot.png').replace('\\', '/')
            
            plt.figure(figsize=(20, 3)) 
            force_plot = shap.plots.force(explanation[0], show=False, matplotlib=True)
            plt.tight_layout()
            plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths['force_plot'] = force_plot_path
            visualization_paths['first_sample_force'] = force_plot_path
            self.logger.debug(f"Force Plot saved to: {force_plot_path}")
        except Exception as e:
            self.logger.error(f"Error generating Force Plot: {str(e)}")
            
            # 备选方案：使用JavaScript输出到HTML，然后尝试捕获
            try:
                # 创建一个临时HTML文件
                html_path = os.path.join(save_dir, 'force_plot_temp.html').replace('\\', '/')
                
                # 使用传统版本的force_plot函数
                force_plot = shap.force_plot(
                    base_value=base_value,
                    shap_values=shap_values[0,:],
                    features=samples[0,:] if samples is not None else None,
                    feature_names=feature_names,
                    figsize=(20, 3),
                    matplotlib=True,
                show=False
            )
                
                plt.tight_layout()
                plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualization_paths['force_plot'] = force_plot_path
                visualization_paths['first_sample_force'] = force_plot_path
                self.logger.debug(f"Fallback Force Plot saved to: {force_plot_path}")
            except Exception as backup_error:
                self.logger.error(f"Fallback force plot generation also failed: {str(backup_error)}")
                
                # 再次备选方案：完全自定义实现力图
                try:
                    # 选择第一个样本
                    sample_idx = 0
                    sample_shap = shap_values[sample_idx]
                    
                    # 对特征重要性进行排序
                    sorted_idx = np.argsort(np.abs(sample_shap))
                    top_features = 20  # 只显示顶部特征
                    sorted_idx = sorted_idx[-top_features:][::-1]
                    
                    # 创建自定义力图
                    plt.figure(figsize=(20, 3))
                    
                    # 基准预测值在中心
                    center_position = 0
                    
                    # 计算正负贡献
                    positive_contributions = sample_shap[sample_shap > 0]
                    negative_contributions = sample_shap[sample_shap < 0]
                    
                    total_positive = positive_contributions.sum() if len(positive_contributions) > 0 else 0
                    total_negative = negative_contributions.sum() if len(negative_contributions) > 0 else 0
                    
                    # 绘制基准线
                    plt.axvline(x=center_position, color='silver', linestyle='-', alpha=0.7)
                    plt.text(center_position, 0.5, f'Base value: {base_value:.3f}', 
                            ha='center', va='center', rotation=90, color='black')
                    
                    # 绘制特征贡献条
                    left_edge = center_position
                    right_edge = center_position
                    
                    # 所有负贡献在左侧
                    for idx in sorted_idx:
                        if sample_shap[idx] < 0:
                            value = sample_shap[idx]
                            name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                            plt.barh(0, value, left=left_edge, color='blue', alpha=0.7)
                            
                            # 添加标签
                            if abs(value) / abs(total_negative) > 0.05:  # 只标注主要贡献
                                plt.text(left_edge + value/2, 0, f'{name}\\n{value:.2f}', 
                                        ha='center', va='center', color='white')
                            
                            left_edge += value
                    
                    # 所有正贡献在右侧
                    for idx in sorted_idx:
                        if sample_shap[idx] > 0:
                            value = sample_shap[idx]
                            name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                            plt.barh(0, value, left=right_edge, color='red', alpha=0.7)
                            
                            # 添加标签
                            if abs(value) / abs(total_positive) > 0.05:  # 只标注主要贡献
                                plt.text(right_edge + value/2, 0, f'{name}\\n+{value:.2f}', 
                                        ha='center', va='center', color='white')
                            
                            right_edge += value
                    
                    # 添加最终预测值
                    final_prediction = base_value + sample_shap.sum()
                    plt.axvline(x=right_edge, color='green', linestyle='-', alpha=0.7)
                    plt.text(right_edge, 0.5, f'Prediction: {final_prediction:.3f}', 
                            ha='center', va='center', rotation=90, color='black')
                    
                    # 隐藏y轴刻度
                    plt.yticks([])
                    plt.xlabel('Feature Contribution')
                    plt.title('SHAP Force Plot - Individual Feature Contributions')
                    
                    # 调整布局和坐标轴
                    plt.xlim(left_edge - abs(total_negative) * 0.1, right_edge + abs(total_positive) * 0.1)
                    plt.tight_layout()
                    
                    # 保存力图
                    plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    visualization_paths['force_plot'] = force_plot_path
                    visualization_paths['first_sample_force'] = force_plot_path
                    self.logger.debug(f"Fully custom Force Plot saved to: {force_plot_path}")
                except Exception as custom_error:
                    self.logger.error(f"Custom force plot generation also failed: {str(custom_error)}")
            except Exception as backup_error:
                self.logger.error(f"Fallback force plot generation also failed: {str(backup_error)}")
        finally:
            if original_float_format is not None:
                try:
                    plt.rcParams['axes.formatter.float_format'] = original_float_format
                    self.logger.debug("Restored original matplotlib float format")
                except KeyError:
                     self.logger.warning("Error restoring original format, might have been removed")
            else:
                plt.rcParams.pop('axes.formatter.float_format', None)
                self.logger.debug("Original format did not exist, attempting to remove potentially added format")
    
    def _save_visualization_stats(self, shap_values, feature_names, save_dir, visualization_paths, top_k=20):
        """Save visualization statistics"""
        try:
            # 打印原始visualization_paths的所有键
            self.logger.info(f"Original visualization_paths contains images: {list(visualization_paths.keys())}")
            
            # 确保所有路径使用正斜杠而不是反斜杠，并移除static前缀用于前端访问
            normalized_paths = {}
            for key, path in visualization_paths.items():
                # 替换Windows风格反斜杠为Web友好的正斜杠，并移除static前缀
                normalized_paths[key] = path.replace('\\', '/').replace('static/', '')
                
            # 打印标准化后的所有路径键值
            self.logger.info(f"Normalized visualization_paths: {normalized_paths}")
                
            with open(os.path.join(save_dir, 'visualization_stats.json').replace('\\', '/'), 'w') as f:
                # 计算全局平均SHAP值绝对值（特征重要性）
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                # 获取top_k个特征的索引
                top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
                
                stats = {
                    "shap_value_range": {
                        "min": float(np.min(shap_values)),
                        "max": float(np.max(shap_values)),
                        "mean": float(np.mean(shap_values)),
                        "abs_mean": float(np.mean(np.abs(shap_values)))
                    },
                    "top_features": [
                        {
                            "name": feature_names[i] if i < len(feature_names) else f"Feature_{i}",
                            "importance": float(mean_abs_shap[i])
                        } 
                        for i in top_indices
                    ],
                    "visualization_paths": normalized_paths
                }
                json.dump(stats, f, indent=2)
                
            # 打印visualization_stats.json的内容
            self.logger.info(f"visualization_paths saved to visualization_stats.json: {normalized_paths}")
        except Exception as e:
            self.logger.error(f"Failed to save statistics data: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def _calculate_statistics(self, shap_values):
        """Calculate statistics for SHAP values"""
        self.logger.debug("_calculate_statistics started")
        
        try:
            # 计算平均特征重要性
            mean_imp = np.mean(shap_values, axis=0)
            
            # 分开药物和蛋白质特征
            drug_features = mean_imp[:self.drug_seq_len]
            protein_features = mean_imp[self.drug_seq_len:]
            
            # 找出贡献最大的位置
            top_drug_idx = np.argmax(np.abs(drug_features))
            top_protein_idx = np.argmax(np.abs(protein_features))
            
            # 分布统计
            positive_features = sum(1 for x in mean_imp if x > 0)
            negative_features = sum(1 for x in mean_imp if x < 0)
            mean_abs_importance = np.mean(np.abs(mean_imp))
            max_abs_importance = np.max(np.abs(mean_imp))
            
            # 确保image_paths存在
            self.logger.debug(f"Checking image_paths before access: Exists={hasattr(self, 'image_paths')}, Type={type(self.image_paths) if hasattr(self, 'image_paths') else 'N/A'}")
            if not hasattr(self, 'image_paths') or self.image_paths is None:
                self.logger.warning("image_paths does not exist or is None, using empty dict instead")
                self.image_paths = {}
            
            # 保存统计结果
            self.results = {
                "images": self.image_paths,
                "statistics": {
                    "positive_features": positive_features,
                    "negative_features": negative_features,
                    "mean_abs_importance": float(mean_abs_importance),
                    "max_abs_importance": float(max_abs_importance),
                    "top_drug_position": int(top_drug_idx),
                    "top_protein_position": int(top_protein_idx)
                }
            }
            
            self.logger.debug("Statistics calculation complete:")
            self.logger.debug(f"Positive impact features: {positive_features} ({positive_features/len(mean_imp)*100:.1f}%)")
            self.logger.debug(f"Negative impact features: {negative_features} ({negative_features/len(mean_imp)*100:.1f}%)")
            self.logger.debug(f"Mean absolute importance: {mean_abs_importance:.5f}")
            self.logger.debug(f"Max absolute importance: {max_abs_importance:.5f}")
            self.logger.debug(f"Top contributing drug position: Drug_Pos_{top_drug_idx} (Importance: {drug_features[top_drug_idx]:.5f})")
            self.logger.debug(f"Top contributing protein position: Protein_Pos_{top_protein_idx} (Importance: {protein_features[top_protein_idx]:.5f})")
            
            self.logger.debug("_calculate_statistics completed")
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise e
    
    async def _notify_progress(self, message, progress):
        """Notify all active connections of progress updates"""
        # 创建正确格式的消息
        message_payload = {
            "type": "shap_status",
            "status": self.status,
            "progress": progress,
            "message": message
        }
        
        # 确保在100%进度时消息包含正确的状态
        if progress == 100:
            message_payload["status"] = "completed"
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message_payload)
                self.logger.debug(f"Sending progress update: {message_payload}")
            except Exception as e:
                self.logger.error(f"Error sending progress update: {str(e)}")
                disconnected.append(connection)
                
        # 移除断开的连接
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

    async def _notify_error(self, message, traceback_str=""):
        """Notify all active connections of error information"""
        self.status = "error" # Set status to error
        self.progress = 0 # Reset progress
        self.error_info["message"] = message
        self.error_info["traceback"] = traceback_str
        
        message_payload = {
            "type": "shap_status", # Match progress type
            "status": "error",
            "progress": 0,
            "message": f"Analysis error: {message}",
            "error": { # Embed error details
                 "message": message,
                 "traceback": traceback_str
            }
        }

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message_payload)
            except Exception as e:
                self.logger.error(f"Error sending error notification: {str(e)}")
                disconnected.append(connection)
        
        # 移除断开的连接
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

    async def poll_status(self):
        """Polls the status of the SHAP analysis"""
        try:
            response = requests.get(f"{API_URL}/api/progress", timeout=3)
            if response.status_code == 200:
                data = response.json()
                print(f"进度API响应: {data}")
        except Exception as e:
            print(f"Error polling status: {e}")