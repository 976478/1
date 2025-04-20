import os
import asyncio
import time
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_manager import ModelManager
from training_manager import TrainingManager
from shap_analyzer import SHAPAnalyzer
from heatmap_generator import HeatmapGenerator
from logger_utils import get_logger, log_execution_time

# 初始化主应用日志
logger = get_logger("Main", "info")
# logger.info("=== MolTrans API 启动 ===")
logger.info("=== MolTrans API Starting ===")

# 数据模型定义
class PredictionRequest(BaseModel):
    smiles: str
    protein_sequence: str
    threshold: Optional[float] = None

class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest]
    
class SHAPRequest(BaseModel):
    sample_ids: Optional[List[int]] = None
    num_samples: int = 30
    features_to_analyze: Optional[List[str]] = None

class TrainingRequest(BaseModel):
    dataset: str = "biosnap"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    save_model: bool = True

class HeatmapRequest(BaseModel):
    smiles: str
    protein_sequence: str
    map_type: str = "attention"
    resolution: str = "high"
    color_map: str = "viridis"

# 创建FastAPI应用
app = FastAPI(title="MolTrans API",
              # description="API for MolTrans drug-protein interaction prediction",
              description="API for MolTrans drug-protein interaction prediction model",
              version="1.0.0")
# logger.info("FastAPI 应用实例已创建")
logger.info("FastAPI application instance created")

# 存储WebSocket连接
active_connections: List[WebSocket] = []

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# logger.debug("CORS 中间件已配置")
logger.debug("CORS middleware configured")

# 设置静态文件目录
@app.on_event("startup")
async def startup_event():
    # logger.info("应用程序启动事件触发")
    logger.info("Application startup event triggered")
    # 确保目录存在
    required_dirs = ["static", "static/images", "saved_models"]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        # logger.debug(f"确保目录存在: {directory}")
        logger.debug(f"Ensuring directory exists: {directory}")
    
    # 初始化管理器
    global model_manager, training_manager, shap_analyzer, heatmap_generator
    # logger.info("初始化组件管理器")
    logger.info("Initializing component managers")
    model_manager = ModelManager(active_connections)
    training_manager = TrainingManager(active_connections)
    shap_analyzer = None  # 等模型加载后初始化
    heatmap_generator = None  # 等模型加载后初始化
    
    # 清理旧图像文件
    image_dir = "static/images"
    cleared_count = 0
    # logger.debug(f"清理旧图像文件: {image_dir}")
    logger.debug(f"Clearing old image files from: {image_dir}")
    for file in os.listdir(image_dir):
        if file.startswith(("shap_", "heatmap_", "loss_", "metrics_")):
            try:
                os.remove(os.path.join(image_dir, file))
                cleared_count += 1
            except Exception as e:
                # logger.warning(f"无法删除文件 {file}: {str(e)}")
                logger.warning(f"Could not delete file {file}: {str(e)}")
    # logger.info(f"已清理 {cleared_count} 个旧图像文件")
    logger.info(f"Cleared {cleared_count} old image files")

    # 添加以下代码来打印路由
    # logger.info("注册的API路由:")
    logger.info("Registered API routes:")
    for route in app.routes:
        # 检查是否有 name 属性，以处理非 APIRoute 的情况 (如 StaticFiles)
        route_name = getattr(route, 'name', 'N/A')
        route_methods = getattr(route, 'methods', 'N/A')
        # logger.debug(f"  路径: {route.path}, 名称: {route_name}, 方法: {route_methods}")
        logger.debug(f"  Path: {route.path}, Name: {route_name}, Methods: {route_methods}")

# 挂载静态文件目录
app.mount("/images", StaticFiles(directory="static/images"), name="images")
# logger.debug("静态文件目录已挂载: /images -> static/images")
logger.debug("Static files directory mounted: /images -> static/images")

# WebSocket连接
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    client = websocket.client.host
    # logger.info(f"WebSocket连接已建立: {client}")
    logger.info(f"WebSocket connection established: {client}")
    
    try:
        # 发送当前状态
        if model_manager.model is not None:
            status_msg = {
                "type": "model_status",
                "status": model_manager.status,
                "progress": model_manager.progress
            }
            await websocket.send_json(status_msg)
            # logger.debug(f"向客户端 {client} 发送模型状态: {model_manager.status}")
            logger.debug(f"Sent model status to client {client}: {model_manager.status}")
        
        if training_manager.status != "idle":
            status_msg = {
                "type": "training_status",
                "status": training_manager.status,
                "progress": training_manager.progress,
                "epoch": training_manager.current_epoch,
                "total_epochs": training_manager.total_epochs
            }
            await websocket.send_json(status_msg)
            # logger.debug(f"向客户端 {client} 发送训练状态: {training_manager.status}")
            logger.debug(f"Sent training status to client {client}: {training_manager.status}")
        
        # 保持连接直到客户端断开
        while True:
            data = await websocket.receive_text()
            # logger.debug(f"从客户端 {client} 接收到消息: {data}")
            logger.debug(f"Received message from client {client}: {data}")
            # 可以处理客户端命令
            if data == "cancel_training":
                training_manager.cancel_training = True
                # logger.info(f"收到客户端 {client} 的训练取消请求")
                logger.info(f"Received training cancellation request from client {client}")
                await websocket.send_json({"message": "Cancellation request received"})
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        # logger.info(f"WebSocket连接已断开: {client}")
        logger.info(f"WebSocket connection disconnected: {client}")

# 模型相关接口
@app.post("/api/load_model")
@log_execution_time
async def load_model_endpoint(model_name: str = "biosnap"):
    model_path = f"saved_models/model_{model_name}.h5"
    # logger.info(f"请求加载模型: {model_name}, 路径: {model_path}")
    logger.info(f"Request to load model: {model_name}, Path: {model_path}")
    
    if not os.path.exists(model_path):
        # logger.error(f"模型文件不存在: {model_path}")
        logger.error(f"Model file not found: {model_path}")
        raise HTTPException(status_code=404, detail="Model not found")
    
    # 后台任务启动模型加载
    asyncio.create_task(model_manager.load_model(model_path))
    # logger.info(f"已创建模型加载后台任务: {model_name}")
    logger.info(f"Background task created for model loading: {model_name}")
    
    return {"status": "loading_started", "model": model_name}

# 添加获取可用模型列表的API端点
@app.get("/api/available_models")
@log_execution_time
async def available_models_endpoint():
    """返回saved_models目录中所有可用的模型文件列表"""
    # logger.info("请求获取可用模型列表")
    logger.info("Request to get available models list")
    try:
        models_dir = "saved_models"
        os.makedirs(models_dir, exist_ok=True)
        
        # 获取目录中所有h5文件
        model_files = [f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".h5")]
        
        # 如果没有找到模型文件，给出友好提示
        if not model_files:
            # logger.warning("未找到可用模型文件")
            logger.warning("No available model files found")
            return {
                "models": [],
                "message": "No models found in the saved_models directory. Please train a model first."
            }
        
        # logger.info(f"找到 {len(model_files)} 个可用模型: {model_files}")
        logger.info(f"Found {len(model_files)} available models: {model_files}")
        return {"models": model_files}
    except Exception as e:
        # logger.error(f"获取模型列表时出错: {str(e)}")
        logger.error(f"Error getting model list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/api/model/status")
async def model_status_endpoint():
    # 确保返回更详细的信息
    response = {
        "status": model_manager.status,
        "progress": model_manager.progress,
        "stage": model_manager.current_stage,
        "timestamp": time.time()  # 添加时间戳以便前端可以判断信息的新旧
    }
    
    # 如果模型已加载，添加模型名称
    if model_manager.status == "ready" and hasattr(model_manager, "model_name"):
        response["current_model"] = model_manager.model_name
        # 添加额外的确认信息
        response["model_loaded"] = True
        response["device"] = str(model_manager.device)
    elif model_manager.status == "loading":
        # 添加更详细的加载状态
        # response["loading_info"] = f"正在加载中，当前进度: {model_manager.progress}%"
        response["loading_info"] = f"Loading in progress, current progress: {model_manager.progress}%"
    elif model_manager.status == "error":
        # 添加错误信息
        # response["error_info"] = getattr(model_manager, "error_message", "未知错误")
        response["error_info"] = getattr(model_manager, "error_message", "Unknown error")
    
    return response

# 添加HTTP轮询进度端点
@app.get("/api/progress")
async def progress_endpoint():
    """
    提供整体进度状态的HTTP轮询端点，用于替代WebSocket连接
    返回当前模型状态、训练状态和各类进度信息
    """
    response = {
        "timestamp": time.time(),
        "model": {
            "status": model_manager.status,
            "progress": model_manager.progress,
            "stage": model_manager.current_stage
        },
        "training": {
            "status": training_manager.status,
            "progress": training_manager.progress,
            "current_epoch": training_manager.current_epoch,
            "total_epochs": training_manager.total_epochs,
            "current_batch": training_manager.current_batch,
            "total_batches": training_manager.total_batches,
            "elapsed_time": time.time() - training_manager.start_time if training_manager.start_time else 0,
            "metrics": training_manager.current_metrics
        }
    }
    
    # 如果有SHAP分析器并且已初始化，添加其状态
    if shap_analyzer is not None:
        response["shap"] = {
            "status": shap_analyzer.status,
            "progress": shap_analyzer.progress
        }
    
    # 如果模型已加载，添加模型名称
    if model_manager.status == "ready" and hasattr(model_manager, "model_name"):
        response["model"]["current_model"] = model_manager.model_name
    
    return response

# 预测接口
@app.post("/api/predict")
@log_execution_time
async def predict_endpoint(request: PredictionRequest):
    # logger.info(f"请求单个预测: SMILES长度={len(request.smiles)}, 蛋白质序列长度={len(request.protein_sequence)}")
    logger.info(f"Request single prediction: SMILES length={len(request.smiles)}, Protein sequence length={len(request.protein_sequence)}")
    
    if model_manager.status != "ready":
        # logger.error("预测请求失败: 模型未就绪")
        logger.error("Prediction request failed: Model not ready")
        raise HTTPException(status_code=400, detail="Model not ready")
    try:
        result = await model_manager.predict(request.smiles, request.protein_sequence, request.threshold)
        # logger.info(f"预测完成: 预测值={result.get('score', 'unknown')}, 原始分数={result.get('raw_score', 'unknown')}")
        logger.info(f"Prediction completed: Prediction={result.get('score', 'unknown')}, Raw score={result.get('raw_score', 'unknown')}")
        return result
    except Exception as e:
        # logger.error(f"预测过程中出错: {str(e)}")
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch_predict")
@log_execution_time
async def batch_predict_endpoint(request: BatchPredictionRequest):
    # logger.info(f"请求批量预测: {len(request.data)}个样本")
    logger.info(f"Request batch prediction: {len(request.data)} samples")
    
    if model_manager.status != "ready":
        # logger.error("批量预测请求失败: 模型未就绪")
        logger.error("Batch prediction request failed: Model not ready")
        raise HTTPException(status_code=400, detail="Model not ready")
    
    results = []
    success_count = 0
    error_count = 0
    
    for idx, item in enumerate(request.data):
        try:
            result = await model_manager.predict(item.smiles, item.protein_sequence, item.threshold)
            results.append(result)
            success_count += 1
        except Exception as e:
            # logger.warning(f"批量预测样本#{idx}出错: {str(e)}")
            logger.warning(f"Batch prediction sample #{idx} error: {str(e)}")
            results.append({"error": str(e)})
            error_count += 1
    
    # logger.info(f"批量预测完成: 成功={success_count}, 失败={error_count}")
    logger.info(f"Batch prediction completed: Success={success_count}, Failed={error_count}")
    return {"results": results}

# SHAP分析接口
@app.post("/api/shap/analyze")
@log_execution_time
async def shap_analyze_endpoint(request: SHAPRequest):
    # logger.info(f"请求SHAP分析: 样本数={request.num_samples}")
    logger.info(f"Request SHAP analysis: Number of samples={request.num_samples}")
    
    if model_manager.status != "ready":
        # logger.error("SHAP分析请求失败: 模型未就绪")
        logger.error("SHAP analysis request failed: Model not ready")
        raise HTTPException(status_code=400, detail="Model not ready")
    
    # 初始化SHAP分析器
    global shap_analyzer
    if shap_analyzer is None:
        # logger.info("创建SHAP分析器")
        logger.info("Creating SHAP Analyzer")
        shap_analyzer = SHAPAnalyzer(model_manager.model, active_connections)
    
    # 后台任务启动SHAP分析
    # logger.info(f"启动SHAP分析后台任务: 样本数={request.num_samples}, 指定样本IDs={request.sample_ids}")
    logger.info(f"Starting SHAP analysis background task: Samples={request.num_samples}, Specific Sample IDs={request.sample_ids}")
    asyncio.create_task(shap_analyzer.analyze(request.num_samples, request.sample_ids))
    
    return {"status": "analysis_started", "num_samples": request.num_samples}

@app.get("/api/shap/status")
async def shap_status_endpoint():
    if shap_analyzer is None:
        return {"status": "not_initialized"}
        
    return {
        "status": shap_analyzer.status,
        "progress": shap_analyzer.progress
    }

@app.get("/api/shap/results")
async def shap_results_endpoint():
    # logger.info("请求SHAP分析结果")
    logger.info("Request SHAP analysis results")
    
    if shap_analyzer is None or shap_analyzer.status != "completed":
        # logger.warning("SHAP结果请求失败: SHAP分析未完成")
        logger.warning("SHAP results request failed: SHAP analysis not completed")
        raise HTTPException(status_code=400, detail="SHAP analysis not completed")
    
    # logger.info("返回SHAP分析结果")
    logger.info("Returning SHAP analysis results")
    return shap_analyzer.results

# 训练接口
@app.post("/api/train")
@log_execution_time
async def train_model_endpoint(request: TrainingRequest):
    # logger.info(f"请求模型训练: 数据集={request.dataset}, 轮次={request.epochs}, 批次大小={request.batch_size}, 学习率={request.learning_rate}")
    logger.info(f"Request model training: Dataset={request.dataset}, Epochs={request.epochs}, Batch Size={request.batch_size}, Learning Rate={request.learning_rate}")
    global training_manager
    
    if training_manager.status in ["training", "preparing"]:
        # logger.warning("训练请求失败: 已有训练正在进行")
        logger.warning("Training request failed: Training already in progress")
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # 后台任务启动训练
    # logger.info(f"启动训练后台任务: 配置={request.dict()}")
    logger.info(f"Starting training background task: Config={request.dict()}")
    asyncio.create_task(training_manager.train_model(request))
    
    return {
        "status": "training_started", 
        "config": request.dict(),
        "websocket_url": "/ws/progress"
    }

# 添加取消训练端点
@app.post("/api/train/cancel")
@log_execution_time
async def cancel_training_endpoint():
    # logger.info("请求取消训练")
    logger.info("Request to cancel training")
    # 首先检查当前训练状态
    current_status = training_manager.status
    current_progress = training_manager.progress
    current_epoch = training_manager.current_epoch
    total_epochs = training_manager.total_epochs
    
    if current_status not in ["training", "preparing"]:
        # logger.warning(f"取消训练请求失败: 当前没有训练在进行，状态为 {current_status}")
        logger.warning(f"Cancel training request failed: No training in progress, current status is {current_status}")
        raise HTTPException(status_code=400, detail="No training in progress to cancel")
    
    # 设置取消标志
    training_manager.cancel_training = True
    # logger.info(f"已设置训练取消标志: 当前状态={current_status}, 进度={current_progress}%, 轮次={current_epoch}/{total_epochs}")
    logger.info(f"Training cancellation flag set: Current Status={current_status}, Progress={current_progress}%, Epoch={current_epoch}/{total_epochs}")
    
    # 返回详细信息，包括当前训练状态
    return {
        "status": "cancellation_requested",
        "message": f"Training cancellation request has been sent. Will cancel at next checkpoint.",
        "current_status": {
            "status": current_status,
            "progress": current_progress,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs
        }
    }

@app.get("/api/train/status")
async def train_status_endpoint():
    return {
        "status": training_manager.status,
        "progress": training_manager.progress,
        "current_epoch": training_manager.current_epoch,
        "total_epochs": training_manager.total_epochs,
        "current_batch": training_manager.current_batch,
        "total_batches": training_manager.total_batches,
        "elapsed_time": time.time() - training_manager.start_time if training_manager.start_time else 0
    }

@app.get("/api/train/metrics")
async def train_metrics_endpoint():
    # logger.info("请求训练指标数据")
    logger.info("Request training metrics data")
    
    if not training_manager.metrics_history:
        # logger.warning("训练指标请求失败: 没有可用的训练指标")
        logger.warning("Training metrics request failed: No training metrics available")
        raise HTTPException(status_code=404, detail="No training metrics available")
    
    # logger.info(f"返回训练指标: {len(training_manager.metrics_history)} 轮")
    logger.info(f"Returning training metrics: {len(training_manager.metrics_history)} epochs")
    return {
        "metrics": training_manager.metrics_history
    }

@app.get("/api/train/plots")
async def train_plots_endpoint():
    # logger.info("请求训练曲线图")
    logger.info("Request training plots")
    
    if not training_manager.plots:
        # logger.warning("训练曲线请求失败: 没有可用的训练曲线图")
        logger.warning("Training plots request failed: No training plots available")
        raise HTTPException(status_code=404, detail="No training plots available")
    
    # logger.info(f"返回训练曲线图: {list(training_manager.plots.keys())}")
    logger.info(f"Returning training plots: {list(training_manager.plots.keys())}")
    return training_manager.plots

# 热力图接口
@app.post("/api/heatmap")
@log_execution_time
async def generate_heatmap_endpoint(request: HeatmapRequest):
    # logger.info(f"请求生成热力图: 类型={request.map_type}, 分辨率={request.resolution}")
    logger.info(f"Request heatmap generation: Type={request.map_type}, Resolution={request.resolution}")
    
    if model_manager.status != "ready":
        # logger.error("热力图请求失败: 模型未就绪")
        logger.error("Heatmap request failed: Model not ready")
        raise HTTPException(status_code=400, detail="Model not ready")
    
    # 初始化热力图生成器
    global heatmap_generator
    if heatmap_generator is None:
        # logger.info("创建热力图生成器")
        logger.info("Creating Heatmap Generator")
        heatmap_generator = HeatmapGenerator(model_manager.model)
    
    try:
        # 生成热力图
        # logger.info(f"生成热力图: SMILES长度={len(request.smiles)}, 蛋白质序列长度={len(request.protein_sequence)}")
        logger.info(f"Generating heatmap: SMILES length={len(request.smiles)}, Protein sequence length={len(request.protein_sequence)}")
        result = await heatmap_generator.generate_heatmap(request)
        # logger.info(f"热力图生成成功: {result}")
        logger.info(f"Heatmap generation successful: {result}")
        return result
    except Exception as e:
        # logger.error(f"生成热力图时出错: {str(e)}")
        logger.error(f"Error generating heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/heatmap/types")
async def get_heatmap_types():
    # logger.debug("请求获取热力图类型信息")
    logger.debug("Requesting heatmap type information")
    return {
        "available_types": ["attention", "interaction", "gradient"],
        "default": "attention",
        "descriptions": {
            "attention": "Shows attention weights between drug and protein sequences",
            "interaction": "Displays interaction strength between each position pair",
            "gradient": "Highlights positions with highest impact on prediction"
        }
    }

# 根路径
@app.get("/")
async def root():
    # logger.debug("访问API根路径")
    logger.debug("Accessing API root path")
    return {
        "message": "MolTrans API is running",
        "docs": "/docs",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # logger.info("主程序入口点执行")
    # logger.info("启动Uvicorn服务器: host=0.0.0.0, port=8000")
    logger.info("Main program entry point executed")
    logger.info("Starting Uvicorn server: host=0.0.0.0, port=8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)