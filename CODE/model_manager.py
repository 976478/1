import os
import time
import asyncio
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 导入MolTrans模型和工具
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder, drug2emb_encoder, protein2emb_encoder

class ModelManager:
    def __init__(self, active_connections: List):
        self.model = None
        self.config = None
        self.status = "not_loaded"
        self.progress = 0
        self.current_stage = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.active_connections = active_connections
        self.best_threshold = 0.5
        self.model_name = None  # 添加模型名称属性
        self.error_message = None  # 添加错误消息属性，用于存储错误信息
        
    async def load_model(self, model_path):
        """异步加载模型，同时发送进度更新"""
        self.status = "loading"
        
        # 从路径中提取模型名称
        try:
            base_name = os.path.basename(model_path)
            if base_name.startswith("model_") and base_name.endswith(".h5"):
                self.model_name = base_name[6:-3]  # 去掉 "model_" 前缀和 ".h5" 后缀
            else:
                self.model_name = os.path.splitext(base_name)[0]
        except:
            self.model_name = "unknown"
        
        # 更新加载阶段: 配置
        self.current_stage = "config"
        self.progress = 10
        await self._notify_progress()
        
        try:
            self.config = BIN_config_DBPE()
            
            # 更新加载阶段: 初始化模型
            self.current_stage = "initialization"
            self.progress = 30
            await self._notify_progress()
            self.model = BIN_Interaction_Flat(**self.config).to(self.device)
            
            # 更新加载阶段: 加载权重
            self.current_stage = "weights"
            self.progress = 50
            await self._notify_progress()
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载模型权重
            try:
                with h5py.File(model_path, 'r') as f:
                    weights_group = f['weights']
                    total_weights = len(weights_group)
                    state_dict = {}
                    
                    for i, name in enumerate(weights_group):
                        self.progress = 50 + int((i / total_weights) * 40)
                        if i % 10 == 0:  # 每加载10个权重更新一次进度
                            await self._notify_progress()
                        
                        state_dict[name] = torch.tensor(weights_group[name][()])
                    
                    # 加载状态字典
                    self.model.load_state_dict(state_dict)
                    
                    # 如果存在阈值信息，也加载
                    if 'training' in f and 'best_threshold' in f['training'].attrs:
                        self.best_threshold = f['training'].attrs['best_threshold']
            except Exception as e:
                self.status = "error"
                self.progress = 0
                self.error_message = f"加载权重失败: {str(e)}"
                await self._notify_progress(error=str(e))
                return None
            
            # 设置模型为评估模式
            self.model.eval()
            
            self.status = "ready"
            self.progress = 100
            await self._notify_progress()
            return self.model
            
        except Exception as e:
            self.status = "error"
            self.progress = 0
            self.error_message = f"模型加载失败: {str(e)}"
            await self._notify_progress(error=str(e))
            return None
    
    async def predict(self, drug_smiles, protein_seq, threshold=None):
        """执行单次预测"""
        if self.status != "ready":
            raise ValueError("Model not ready")
        
        # 预处理
        d_v, input_mask_d = drug2emb_encoder(drug_smiles)
        p_v, input_mask_p = protein2emb_encoder(protein_seq)
        
        # 转换为张量
        d = torch.tensor([d_v]).long().to(self.device)
        p = torch.tensor([p_v]).long().to(self.device)
        d_mask = torch.tensor([input_mask_d]).long().to(self.device)
        p_mask = torch.tensor([input_mask_p]).long().to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            score = self.model(d, p, d_mask, p_mask)
            m = torch.nn.Sigmoid()
            proba = torch.squeeze(m(score)).item()
        
        # 应用阈值
        threshold = threshold or self.best_threshold
        pred_label = 1 if proba >= threshold else 0
        
        return {
            "probability": proba,
            "prediction": pred_label,
            "threshold_used": threshold,
            "drug_length": int(sum(input_mask_d)),
            "protein_length": int(sum(input_mask_p))
        }
    
    async def _notify_progress(self, error=None):
        """发送进度更新到WebSocket连接的客户端"""
        message = {
            "type": "model_status",
            "status": self.status,
            "stage": self.current_stage,
            "progress": self.progress
        }
        
        # 如果模型名称存在，添加到消息中
        if self.model_name:
            message["model_name"] = self.model_name
        
        if error:
            message["error"] = error
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # 移除断开的连接
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)