import requests
import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import time
import websocket
import json
import threading
import os
from streamlit.components.v1 import html
from datetime import datetime

# 配置后端API地址
API_URL = "http://127.0.0.1:8000"

# 显示模型状态的函数
def display_model_status():
    # 添加状态提示框，根据模型状态显示不同内容
    status = st.session_state.model_status
    
    if status == "ready":
        model_name = st.session_state.current_model_name
        st.success(f"✅ Model loaded and ready: {model_name}")
    elif status == "loading":
        st.info("⏳ Model is loading...")
        # 创建一个空元素来显示进度条，稍后可以更新它
        progress_placeholder = st.empty()
        # 如果有进度信息，显示进度条
        if hasattr(st.session_state, 'progress') and st.session_state.progress > 0:
            progress_placeholder.progress(st.session_state.progress / 100)
    elif status == "error":
        st.error("❌ Model loading failed")
        if 'error_message' in st.session_state:
            st.error(f"Error details: {st.session_state.error_message}")
    else:
        st.warning("⚠️ No model loaded, please select and load a model first")

# 会话状态初始化 - 移到文件最开始，确保在使用前初始化所有变量
if 'model_status' not in st.session_state:
    st.session_state.model_status = "not_loaded"
    
if 'training_status' not in st.session_state:
    st.session_state.training_status = "idle"
    
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
    
if 'training_epoch' not in st.session_state:
    st.session_state.training_epoch = 0
    
if 'training_total_epochs' not in st.session_state:
    st.session_state.training_total_epochs = 0
    
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = {}
    
if 'shap_status' not in st.session_state:
    st.session_state.shap_status = "not_started"
    
if 'show_shap_results' not in st.session_state:
    st.session_state.show_shap_results = False
    
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False
    
# 添加轮询设置
if 'last_poll_time' not in st.session_state:
    st.session_state.last_poll_time = 0
    
if 'polling_interval' not in st.session_state:
    st.session_state.polling_interval = 1.0  # 轮询间隔(秒)

# 添加会话状态变量用于标记需要发送训练请求
if 'needs_train_request' not in st.session_state:
    st.session_state.needs_train_request = False
    
if 'train_params' not in st.session_state:
    st.session_state.train_params = {}

# 添加连接状态变量初始化
if 'connected' not in st.session_state:
    st.session_state.connected = True

# WebSocket消息初始化
if 'ws_messages' not in st.session_state:
    st.session_state.ws_messages = []

if 'ws_message_queue' not in st.session_state:
    st.session_state.ws_message_queue = []

if 'cancel_requested' not in st.session_state:
    st.session_state.cancel_requested = False

if 'current_model_name' not in st.session_state:
    st.session_state.current_model_name = ""

# 页面初始化时立即轮询一次后端状态，确保页面刷新后能获取最新的模型状态
try:
    # 轮询后端状态
    response = requests.get(f"{API_URL}/api/model/status", timeout=2)
    if response.status_code == 200:
        data = response.json()
        st.session_state.model_status = data.get("status", "not_loaded")
        # 如果模型已就绪，记录当前模型名称
        if data.get("status") == "ready" and "current_model" in data:
            st.session_state.current_model_name = data["current_model"]
    # 同时检查训练和SHAP状态
    response = requests.get(f"{API_URL}/api/progress", timeout=3)
    if response.status_code == 200:
        data = response.json()
        # 更新训练状态
        training_info = data.get("training", {})
        st.session_state.training_status = training_info.get("status", st.session_state.training_status)
        st.session_state.training_progress = training_info.get("progress", st.session_state.training_progress)
        # 更新SHAP状态
        if "shap" in data:
            shap_info = data.get("shap", {})
            st.session_state.shap_status = shap_info.get("status", "not_started")
except Exception as e:
    # 出错时不更新状态，保持默认值
    print(f"初始化轮询状态时出错: {e}")

# 添加轮询功能
def poll_status(force=False):
    """轮询后端状态的函数，替代WebSocket"""
    current_time = time.time()
    
    # 如果强制轮询或者已经到了轮询时间
    if force or (current_time - st.session_state.last_poll_time >= st.session_state.polling_interval):
        # 首先单独检查训练状态，因为这是最关键的
        try:
            # 首先尝试训练状态API
            training_response = requests.get(f"{API_URL}/api/train/status", timeout=2)
            if training_response.status_code == 200:
                training_data = training_response.json()
                # 更新训练状态
                st.session_state.training_status = training_data.get("status", st.session_state.training_status)
                st.session_state.training_progress = training_data.get("progress", st.session_state.training_progress)
                st.session_state.training_epoch = training_data.get("current_epoch", st.session_state.training_epoch)
                st.session_state.training_total_epochs = training_data.get("total_epochs", st.session_state.training_total_epochs)
                
                # 如果正在训练，减少轮询间隔
                if st.session_state.training_status in ["training", "preparing", "cancelling"]:
                    st.session_state.polling_interval = 0.5  # 训练中更频繁地轮询
                
                # 更新最后轮询时间
                st.session_state.last_poll_time = current_time
                # 标记已连接
                st.session_state.connected = True
        except Exception as e:
            print(f"轮询训练状态时出错: {e}")
            # 这里不要重置训练状态，保持之前的状态
            st.session_state.connected = False
        
        # 然后尝试常规进度API
        try:
            # 调用新的进度API
            response = requests.get(f"{API_URL}/api/progress", timeout=3)
            if response.status_code == 200:
                data = response.json()
                
                # 添加模型名跟踪
                if 'current_model_name' not in st.session_state:
                    st.session_state.current_model_name = ""

                # 更新模型状态
                model_info = data.get("model", {})
                st.session_state.model_status = model_info.get("status", "not_loaded")
                
                # 如果模型状态为ready，保存当前模型名称
                if model_info.get("status") == "ready" and "current_model" in data.get("model", {}):
                    st.session_state.current_model_name = data["model"]["current_model"]
                
                # 更新训练状态 (进度API返回的数据更全面)
                training_info = data.get("training", {})
                st.session_state.training_status = training_info.get("status", st.session_state.training_status)
                st.session_state.training_progress = training_info.get("progress", st.session_state.training_progress)
                st.session_state.training_epoch = training_info.get("current_epoch", st.session_state.training_epoch)
                st.session_state.training_total_epochs = training_info.get("total_epochs", st.session_state.training_total_epochs)
                st.session_state.training_metrics = training_info.get("metrics", {})
                
                # 更新SHAP状态，但不更新进度
                if "shap" in data:
                    shap_info = data.get("shap", {})
                    st.session_state.shap_status = shap_info.get("status", "not_started")
                    # 不更新进度值
                
                # 更新轮询时间
                st.session_state.last_poll_time = current_time
                # 标记已连接
                st.session_state.connected = True
                
                # 在关键状态变化时调整轮询频率
                if st.session_state.training_status in ["training", "preparing", "cancelling"]:
                    st.session_state.polling_interval = 0.5  # 训练中更频繁地轮询
                else:
                    st.session_state.polling_interval = 2.0  # 空闲时减少轮询频率
            
        except Exception as e:
            print(f"轮询进度状态时出错: {e}")
            # 标记连接失败
            st.session_state.connected = False

    # 修改策略：当后端连接失败时，不要自动增加进度，这会误导用户
    # 仅在最后一次成功连接后的一段时间内自动增加进度
    if not getattr(st.session_state, 'connected', True) and getattr(st.session_state, 'training_status', 'idle') in ['training', 'preparing']:
        # 如果连接断开但仍在训练，确保用户可以尝试取消训练
        # 将最后轮询时间设为当前时间，以保持定期刷新
        st.session_state.last_poll_time = current_time

# 检查API连接状态
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            return True, response.json()
        return False, {"message": f"API返回错误: {response.status_code}"}
    except Exception as e:
        return False, {"message": f"无法连接到API: {str(e)}"}

# 在加载按钮点击时调用的函数
def load_model_click(model_name):
    try:
        # 发送加载请求
        with st.spinner("Loading model..."):
            response = requests.post(f"{API_URL}/api/load_model?model_name={model_name}")
            if response.status_code == 200:
                st.success(f"✅ Loading model: {model_name}")
                # 更新状态
                st.session_state.model_status = "loading"
                
                # 立即轮询一次获取最新状态
                poll_status(force=True)
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                st.error(f"❌ Loading failed: {error_msg}")
                # 记录错误信息
                st.session_state.error_message = error_msg
                if "not found" in error_msg:
                    st.warning("⚠️ Model file does not exist, please train a model first or check the filename")
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")
        # 记录错误信息
        st.session_state.error_message = str(e)

# 设置页面主题和布局
st.set_page_config(
    page_title="MolTrans Model Analysis & Visualization", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/moltrans',
        'Report a bug': 'https://github.com/yourusername/moltrans/issues',
        'About': 'This application shows model statistics and SHAP interpretability analysis for drug-protein interaction prediction'
    }
)

# 自定义页面样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E3F2FD;
    }
    .card {
        background-color: #F5F7FA;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .info-text {
        color: #555;
        font-size: 16px;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        padding-top: 1rem;
        margin-top: 2rem;
    }
    .plot-container {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        border: 1px solid #E3F2FD;
    }
    .warning-box {
        background-color: #FFF8E1;
        border-left: 4px solid #FFB300;
        padding: 15px;
        border-radius: 0 5px 5px 0;
        margin-bottom: 20px;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
    }
    .grid-item {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        border: 1px solid #E3F2FD;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .grid-item-title {
        text-align: center;
        font-weight: bold;
        color: #1565C0;
        margin-bottom: 10px;
    }
    .shap-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-top: 20px;
    }
    @media (max-width: 1200px) {
        .shap-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 768px) {
        .shap-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# 创建一个有颜色的顶部条
st.markdown("""
<div style="background-color:#E3F2FD;padding:10px;border-radius:5px;margin-bottom:20px;">
    <h3 style="color:#1565C0;margin:0;text-align:center;">Drug-Protein Interaction Prediction Analysis Tool</h3>
</div>
""", unsafe_allow_html=True)

# 轮询状态（强制刷新，确保获取最新状态）
poll_status(force=True)

# 添加SHAP结果获取函数
def get_shap_results():
    """从后端获取SHAP分析结果"""
    if st.session_state.shap_status == "completed" and not 'shap_results' in st.session_state:
        try:
            response = requests.get(f"{API_URL}/api/shap/results")
            if response.status_code == 200:
                st.session_state.shap_results = response.json()
                print("成功获取SHAP结果")
                return True
            else:
                print(f"获取SHAP结果失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"SHAP结果获取错误: {str(e)}")
            return False
    return 'shap_results' in st.session_state

# 修改简化训练请求处理部分，添加更多错误处理
# 简化训练请求处理，只检查状态和参数
# 如果处于preparing状态并且有训练参数，则发送请求
if (st.session_state.training_status == "preparing" 
        and 'train_params' in st.session_state 
        and st.session_state.train_params):
    
    # 显示正在发送请求提示，但不阻塞UI
    st.sidebar.info("⏳ 正在发送训练请求...")
    
    try:
        # 执行请求但不等待结果
        import threading
        
        def send_request():
            try:
                response = requests.post(
                    f"{API_URL}/api/train", 
                    json=st.session_state.train_params,
                    timeout=10
                )
                # 在这里不处理响应结果，由轮询逻辑处理
                print(f"训练请求发送完成，状态码: {response.status_code}")
            except Exception as e:
                print(f"训练请求发送错误: {e}")
        
        # 在后台线程发送请求
        thread = threading.Thread(target=send_request)
        thread.daemon = True
        thread.start()
        
        # 立即更新状态为training并清空参数
        st.session_state.training_status = "training"
        st.session_state.train_params = {}
        
    except Exception as e:
        print(f"启动训练请求线程时出错: {e}")
        # 记录错误但保持preparing状态，允许用户重试

# 在页面顶部添加训练状态和紧急重置按钮
if getattr(st.session_state, 'training_status', 'idle') in ['training', 'preparing', 'cancelling']:
    # 不显示训练状态信息和进度条
    pass

# WebSocket消息处理函数
def on_message(ws, message):
    try:
        data = json.loads(message)
        message_type = data.get("type", "")
        
        if message_type == "model_status":
            st.session_state.model_status = data.get("status", "unknown")
            st.session_state.model_progress = data.get("progress", 0)
        elif message_type == "training_status":
            st.session_state.training_status = data.get("status", "unknown")
            st.session_state.training_progress = data.get("progress", 0)
            st.session_state.training_epoch = data.get("epoch", 0)
            st.session_state.training_total_epochs = data.get("total_epochs", 0)
            if "latest_metrics" in data:
                st.session_state.training_metrics = data["latest_metrics"]
                
            # 处理取消训练的响应
            if data.get("status") == "cancelled":
                st.session_state.training_status = "cancelled"
                # 重置训练状态
                st.session_state.cancel_requested = False
        elif message_type == "shap_status":
            st.session_state.shap_status = data.get("status", "unknown")
            if data.get("status") == "completed":
                st.session_state.show_shap_results = True
        
        # 存储消息历史
        st.session_state.ws_messages.append(data)
    except Exception as e:
        print(f"WebSocket消息处理错误: {e}")

def on_error(ws, error):
    print(f"WebSocket错误: {error}")
    st.session_state.ws_connected = False

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket连接关闭: {close_status_code} - {close_msg}")
    st.session_state.ws_connected = False

def on_open(ws):
    print("WebSocket连接已打开")
    st.session_state.ws_connected = True

# 修改WebSocket连接函数
def connect_websocket():
    # 禁用WebSocket连接，使用HTTP轮询代替
    print("WebSocket连接已禁用，使用HTTP轮询代替")
    st.session_state.ws_connected = False
    return None
    
# 添加一个处理WebSocket消息队列的函数
def process_ws_messages():
    """处理WebSocket消息队列中的消息"""
    if 'ws_message_queue' in st.session_state and st.session_state.ws_message_queue:
        # 获取队列中的所有消息
        messages = st.session_state.ws_message_queue.copy()
        # 清空队列
        st.session_state.ws_message_queue = []
        
        # 处理每条消息
        for data in messages:
            message_type = data.get("type", "")
            
            if message_type == "model_status":
                    st.session_state.model_status = data.get("status", "unknown")
                    st.session_state.model_progress = data.get("progress", 0)
            elif message_type == "training_status":
                st.session_state.training_status = data.get("status", "unknown")
                st.session_state.training_progress = data.get("progress", 0)
                st.session_state.training_epoch = data.get("epoch", 0)
                st.session_state.training_total_epochs = data.get("total_epochs", 0)
                if "latest_metrics" in data:
                    st.session_state.training_metrics = data["latest_metrics"]
                    
                # 处理取消训练的响应
                if data.get("status") == "cancelled":
                    st.session_state.training_status = "cancelled"
                    # 重置训练状态
                    st.session_state.cancel_requested = False
            elif message_type == "shap_status":
                st.session_state.shap_status = data.get("status", "unknown")
                if data.get("status") == "completed":
                    st.session_state.show_shap_results = True
            
            # 存储消息历史
            if 'ws_messages' not in st.session_state:
                st.session_state.ws_messages = []
            st.session_state.ws_messages.append(data)

# 侧边栏
with st.sidebar:
    st.markdown('<div style="text-align:center;font-size:1.5rem;font-weight:bold;color:#1565C0;margin-bottom:20px;">Configuration Panel</div>', unsafe_allow_html=True)
    
    # 紧急重置按钮 - 仅在需要时显示
    if st.session_state.training_status in ["training", "preparing", "cancelling"]:
        # 不显示紧急重置按钮
        pass
    
    # 设置固定的颜色主题（使用蓝色主题）
    primary_color = "#1565C0"
    secondary_color = "#E3F2FD"
    
    # 应用固定主题颜色
    st.markdown(f"""
    <style>
        /* 全局元素 */
        .main-header {{ color: {primary_color} !important; }}
        .sub-header {{ color: {primary_color} !important; border-bottom: 2px solid {secondary_color} !important; }}
        .stButton button {{ background-color: {primary_color} !important; color: white !important; }}
        
        /* 侧边栏样式 */
        div[data-testid="stSidebarNav"] li div a {{ color: {primary_color} !important; }}
        
        /* 标题和卡片元素 */
        .grid-item-title {{ color: {primary_color} !important; }}
        
        /* 进度条颜色 */
        .stProgress > div > div > div > div {{
            background-color: {primary_color} !important;
        }}
        
        /* 选择器颜色 */
        .stSelectbox label, .stMultiSelect label {{ color: {primary_color} !important; }}
        
        /* 链接颜色 */
        a {{ color: {primary_color} !important; }}
        
        /* 顶部条和卡片 */
        div.card {{ border-left: 3px solid {primary_color} !important; }}
        
        /* 网格项 */
        .grid-item {{ border-top: 3px solid {primary_color} !important; }}
        
        /* 选项卡指示器 */
        button[data-baseweb="tab"] {{ color: {primary_color} !important; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ 
            color: {primary_color} !important;
            border-bottom-color: {primary_color} !important;
        }}
        
        /* 警告框 */
        .warning-box {{ border-left: 4px solid {primary_color} !important; }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr style="margin-top:0;margin-bottom:20px;">', unsafe_allow_html=True)
    
    # 连接状态
    api_status, api_info = check_api_status()
    if api_status:
        st.success("✅ API service is normal")
    else:
        st.error(f"❌ API connection failed: {api_info['message']}")
        # 添加帮助信息
        if not getattr(st.session_state, 'connected', True) and getattr(st.session_state, 'training_status', 'idle') in ['training', 'preparing', 'cancelling']:
            st.warning("""
            ⚠️ **Backend connection lost, but training may still be in progress**
            
            Possible solutions:
            1. Check if the backend service is still running (python main.py)
            2. If the backend has stopped, you can click "Force Reset Training Status"
            3. If the backend is still running, try refreshing the page or restarting the frontend
            """)

    # 刷新状态按钮
    if st.button("🔄 Refresh Status", use_container_width=True):
        try:
            with st.spinner("Refreshing status..."):
                # 检查模型状态
                response = requests.get(f"{API_URL}/api/model/status")
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.model_status = data.get("status", "unknown")
                    st.session_state.model_progress = data.get("progress", 0)
                    
                    # 如果模型已就绪并且包含current_model字段，更新当前模型名称
                    if data.get("status") == "ready" and "current_model" in data:
                        st.session_state.current_model_name = data["current_model"]
                
                # 检查训练状态
                if st.session_state.training_status in ["training", "preparing"]:
                    train_response = requests.get(f"{API_URL}/api/train/status")
                    if train_response.status_code == 200:
                        train_data = train_response.json()
                        st.session_state.training_status = train_data.get("status", "unknown")
                        st.session_state.training_progress = train_data.get("progress", 0)
                
                st.success("✅ Status refreshed!")
                # 触发页面刷新
                st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ Refresh failed: {str(e)}")
    
    # 功能区域
    st.markdown('<div style="font-weight:bold;margin-top:20px;margin-bottom:10px;">Model Operations</div>', unsafe_allow_html=True)
    
    # 添加状态指示器
    if st.session_state.training_status in ["training", "preparing"]:
        st.markdown("""
        <div style="display:flex;align-items:center;margin-bottom:10px;background-color:#FFF8E1;padding:5px;border-radius:4px;">
            <div style="margin-right:10px;">⚠️</div>
            <div style="font-size:0.9rem;">Training in progress, some operations may be limited</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 模型加载区域
    with st.expander("Model Loading Options", expanded=True):
        # 尝试获取saved_models目录下的所有模型文件
        available_models = []
        try:
            response = requests.get(f"{API_URL}/api/available_models")
            if response.status_code == 200:
                available_models = response.json().get("models", [])
            else:
                st.warning("⚠️ Unable to get available model list")
        except Exception as e:
                st.warning(f"⚠️ Error getting model list: {str(e)}")
        
        # 如果获取到了模型列表，显示下拉选择框
        if available_models:
            selected_model_file = st.selectbox(
                "Select model to load",
                options=available_models,
                format_func=lambda x: x.replace("model_", "").replace(".h5", "")
            )
            # 提取模型名称，去掉前缀和后缀
            model_name = selected_model_file.replace("model_", "").replace(".h5", "")
        else:
            # 如果没有获取到模型列表，显示文本输入框
            model_name = st.text_input("Model name", value="biosnap")
            st.info("💡 Enter model name, system will try to load saved_models/model_[model_name].h5 file")
        
        if st.button("Load Model", use_container_width=True):
            load_model_click(model_name)
    
    # 训练区域
    with st.expander("🚀 Training Settings"):
        # 保留训练设置选项
        dataset_options = ["biosnap", "bindingdb", "davis"]
        selected_dataset = st.selectbox("Dataset", dataset_options, index=0)
        
        epochs = st.slider("Training Epochs", min_value=1, max_value=20, value=5)
        batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=32, step=8)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            format_func=lambda x: f"{x:.4f}"
        )
        save_model = st.checkbox("Save Model", value=True)
        
        # 只保留开始训练按钮，移除停止训练按钮
        if st.button("Start Training", use_container_width=True):
            # 检查是否已经在训练中
            training_in_progress = st.session_state.training_status in ["training", "preparing", "cancelling"]
            if training_in_progress:
                st.warning("⚠️ Training is already in progress")
            else:
                try:
                    # 安全地设置状态变量
                    st.session_state.training_status = "preparing"
                    st.session_state.training_progress = 0
                    st.session_state.training_epoch = 0
                    st.session_state.training_total_epochs = epochs
                    
                    # 准备训练参数
                    train_params = {
                        "dataset": selected_dataset,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "save_model": save_model
                    }
                    
                    # 直接发送训练请求
                    st.info("⏳ Sending training request...")
                    
                    response = requests.post(
                        f"{API_URL}/api/train", 
                        json=train_params,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        st.success("✅ Training started!")
                        # 保存参数以便在状态中显示
                        st.session_state.train_params = train_params
                        # 立即轮询一次获取最新状态
                        poll_status(force=True)
                    else:
                        st.error(f"❌ Training request failed: {response.json().get('detail', 'Unknown error')}")
                        # 恢复状态
                        st.session_state.training_status = "idle"
                    
                except Exception as e:
                    st.error(f"During training: {str(e)}")
                    # 恢复状态以防止卡住
                    st.session_state.training_status = "idle"
    
    # 预测输入
    with st.expander("🔍 Prediction Settings"):
        st.markdown('<div style="font-weight:bold;margin-top:10px;margin-bottom:10px;">Input Data</div>', unsafe_allow_html=True)
        drug_smiles = st.text_area("Drug SMILES", value="CC(=O)NC1=CC=C(O)C=C1", height=80)
        protein_seq = st.text_area("Protein Sequence", value="MEIGYLTDEKT", height=80)
        threshold = st.slider("Prediction Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    # SHAP分析设置
    with st.expander("📊 SHAP Analysis Settings"):
        # 分析参数设置
        st.markdown("<div style='margin-bottom:10px;font-weight:bold;'>Analysis Parameters</div>", unsafe_allow_html=True)
        num_samples = st.slider("Number of Samples", min_value=5, max_value=100, value=30)
        
        # 添加分割线
        st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
        
        # 状态显示和操作区域
        status_col1, status_col2 = st.columns([3, 1])
        
        with status_col1:
            # 当前状态显示
            if st.session_state.model_status != "ready":
                st.warning("⚠️ Please load model before SHAP analysis")
            elif st.session_state.shap_status == "running":
                st.info("⏳ Analysis in progress...")
            elif st.session_state.shap_status == "completed":
                st.success("✅ Analysis completed")
            elif st.session_state.shap_status == "error":
                st.error("❌ Analysis error")
        
        with status_col2:
            # 操作按钮
            shap_disabled = st.session_state.model_status != "ready"
            start_button = st.button(
                "Start Analysis", 
                use_container_width=True,
                disabled=shap_disabled
            )
        
        # 处理按钮点击
        if start_button:
            with st.spinner("Preparing SHAP analysis..."):
                try:
                    # 添加检查：确保模型已加载
                    if st.session_state.model_status != "ready":
                        st.error("❌ Model not loaded, cannot perform SHAP analysis")
                    else:
                        # 准备请求参数
                        payload = {
                            "num_samples": num_samples
                        }
                        
                        # 发送请求
                    response = requests.post(f"{API_URL}/api/shap/analyze", json=payload)
                        
                        # 处理响应
                    if response.status_code == 200:
                            # 更新状态
                            st.session_state.shap_status = "running"
                            
                            # 清除之前的结果
                            if 'shap_results' in st.session_state:
                                del st.session_state.shap_results
                                
                            st.success("✅ SHAP analysis started, please switch to 'SHAP Analysis' tab to view progress!")
                            
                            # 自动切换到SHAP标签页
                            js = f"""
                            <script>
                                window.parent.document.querySelectorAll('button[data-baseweb="tab"]')[1].click();
                            </script>
                            """
                            html(js)
                    else:
                            error_msg = response.json().get('detail', 'Unknown error')
                            st.error(f"❌ SHAP analysis failed: {error_msg}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
        
        # 取消按钮 - 仅在分析进行中显示
        if st.session_state.shap_status == "running":
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            if st.button("⚠️ Cancel ongoing analysis", key="cancel_shap_in_sidebar", type="secondary"):
                try:
                    requests.post(f"{API_URL}/api/shap/cancel")
                    st.success("Cancel request sent")
                except Exception as e:
                    st.error(f"Cancel request failed: {str(e)}")
        

    
    # 热力图设置
    with st.expander("🔥 Heatmap Settings"):
        map_types = ["attention", "interaction", "gradient"]
        selected_map_type = st.selectbox("Heatmap Type", map_types)
        
        resolutions = ["low", "medium", "high"]
        selected_resolution = st.selectbox("Resolution", resolutions, index=1)
        
        color_maps = ["viridis", "plasma", "inferno", "magma", "cividis", "YlGnBu", "RdBu"]
        selected_color_map = st.selectbox("Color Map", color_maps)
        
        if st.button("Generate Heatmap", use_container_width=True):
            with st.spinner("Generating heatmap..."):
                try:
                    payload = {
                        "smiles": drug_smiles,
                        "protein_sequence": protein_seq,
                        "map_type": selected_map_type,
                        "resolution": selected_resolution,
                        "color_map": selected_color_map
                    }
                    response = requests.post(f"{API_URL}/api/heatmap", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.heatmap_result = result
                        st.success("✅ Heatmap generated successfully!")
                    else:
                        st.error(f"❌ Heatmap generation failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
    
    # 在高级设置中添加调试模式
    with st.expander("⚙️ Advanced Settings"):
        custom_api_url = st.text_input("API URL", API_URL)
        if custom_api_url != API_URL:
            API_URL = custom_api_url
            st.success(f"API URL updated: {API_URL}")
        
        # 添加调试模式开关
        debug_mode = st.checkbox("Debug Mode", value=False)
        if debug_mode:
            st.write("### Current Session State Variables")
            # 显示所有会话状态变量
            state_dict = {k: v for k, v in st.session_state.items()}
            st.json(state_dict)
            
            st.write("### Backend API Status")
            try:
                # 获取当前后端状态
                response = requests.get(f"{API_URL}/api/progress", timeout=3)
                if response.status_code == 200:
                    st.json(response.json())
                else:
                    st.error(f"Failed to get backend status: {response.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to backend API: {e}")
        
        # 添加重置所有会话状态选项
        if st.button("Reset All States", type="primary"):
            for key in list(st.session_state.keys()):
                if key != "debug_mode":  # 保留调试模式设置
                    del st.session_state[key]
            st.success("All state variables have been reset! Page will refresh...")
            st.experimental_rerun()
    
    # 显示当前模型状态
    st.markdown('<hr style="margin-top:20px;margin-bottom:20px;">', unsafe_allow_html=True)
    
    # 模型状态
    status_container = st.container()
    with status_container:
        # 模型加载状态
        display_model_status()
        
        # 训练状态
        if st.session_state.training_status == "training" or st.session_state.training_status == "preparing":
            # 不显示帮助信息、取消训练按钮和进度条
            pass
        elif st.session_state.training_status == "completed":
            # 不显示已完成状态
            pass
        elif st.session_state.training_status == "cancelled":
            # 不显示已取消状态
            pass
        elif st.session_state.training_status == "cancelling":
            # 不显示取消中状态
            pass
        
        

    # 诊断工具
    with st.expander("🔧 Diagnostic Tools"):
        if st.button("📊 Check Backend Status"):
            try:
                # 测试基本连接
                root_response = requests.get(f"{API_URL}/")
                st.write(f"API Root Path: {'✅' if root_response.status_code == 200 else '❌'} ({root_response.status_code})")
                
                # 检查模型状态端点
                status_response = requests.get(f"{API_URL}/api/model/status")
                st.write(f"Model Status API: {'✅' if status_response.status_code == 200 else '❌'} ({status_response.status_code})")
                if status_response.status_code == 200:
                    st.json(status_response.json())
                    
                # 检查目录结构
                if os.path.exists("saved_models"):
                    st.write("saved_models directory: ✅")
                    models = os.listdir("saved_models")
                    st.write(f"Available model files: {models}")
                else:
                    st.write("saved_models directory: ❌ (not exist)")
                    
                # 检查静态文件目录
                if os.path.exists("static/images"):
                    st.write("static/images directory: ✅")
                else:
                    st.write("static/images directory: ❌ (not exist)")
                    
            except Exception as e:
                st.error(f"Error during diagnosis: {str(e)}")

        # 移除训练取消测试按钮
        if st.session_state.training_status in ["training", "preparing"]:
            # 不显示测试按钮
            pass

# 主界面内容区 - 使用多个标签页
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Prediction", "🔍 SHAP Analysis", "🔥 Heatmap", "⚙️ Training Curves"])

# 标签页1: 模型统计和预测
with tab1:
    st.markdown('<div class="sub-header">Model Performance & Prediction</div>', unsafe_allow_html=True)
    
    # 预测区域
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#1565C0;margin-top:0;">Single Prediction</h4>', unsafe_allow_html=True)
    
    pred_col1, pred_col2 = st.columns([3, 1])
    
    with pred_col1:
        # 显示当前选择的输入
        st.markdown(f"**Drug SMILES**: `{drug_smiles[:50]}...`" if len(drug_smiles) > 50 else f"**Drug SMILES**: `{drug_smiles}`")
        st.markdown(f"**Protein Sequence**: `{protein_seq[:50]}...`" if len(protein_seq) > 50 else f"**Protein Sequence**: `{protein_seq}`")
    
    with pred_col2:
        # 添加预测按钮
        predict_button = st.button("Start Prediction", use_container_width=True)
        
    # 设置预测结果区域
    result_container = st.container()
    
    # 如果点击了预测按钮
    if predict_button:
        if st.session_state.model_status != "ready":
            result_container.warning("⚠️ Please load a model before prediction")
        else:
            try:
                with result_container.status("Predicting...") as status:
                    # 准备预测数据
                    predict_data = {
                        "smiles": drug_smiles,
                        "protein_sequence": protein_seq,
                        "threshold": threshold
                    }
                    
                    # 发送预测请求
                    response = requests.post(
                        f"{API_URL}/api/predict",
                        json=predict_data
                    )
                    
                    # 处理响应
                    if response.status_code == 200:
                        prediction_result = response.json()
                        
                        # 更新状态
                        status.update(label="✅ Prediction completed!", state="complete")
                        
                        # 显示预测结果
                        result_container.markdown("### Prediction Result")
                        
                        # 创建结果列
                        res_col1, res_col2 = result_container.columns(2)
                        
                        # 预测概率和绑定情况
                        probability = prediction_result.get("probability", 0)
                        prediction_label = prediction_result.get("prediction", 0)

                        # 在第一列显示概率值
                        res_col1.metric(
                            label="Binding Probability", 
                            value=f"{probability:.4f}",
                            delta=f"{'Above' if probability > threshold else 'Below'} threshold ({threshold})"
                        )
                        
                        # 在第二列显示绑定结果
                        if prediction_label == 1:
                            res_col2.success("**Prediction**: Likely to bind ✅")
                        else:
                            res_col2.error("**Prediction**: Unlikely to bind ❌")
                        
                        # 显示置信度指标
                        if "confidence" in prediction_result:
                            result_container.info(f"**Confidence**: {prediction_result['confidence']:.2f}")
                            
                        # 显示其他可能的元数据
                        if "metadata" in prediction_result:
                            result_container.json(prediction_result["metadata"])
                    else:
                        # 显示错误信息
                        error_msg = response.json().get("detail", "Unknown error")
                        status.update(label=f"❌ Prediction failed: {error_msg}", state="error")
            except Exception as e:
                result_container.error(f"❌ Connection error: {str(e)}")
    
    # 关闭卡片div
    st.markdown('</div>', unsafe_allow_html=True)

# 标签页2: SHAP分析
with tab2:
    # 添加调试信息(可选)
    if debug_mode:
        st.write("SHAP Status:", st.session_state.shap_status)
        st.write("Show Results Flag:", st.session_state.show_shap_results)
        if 'shap_results' in st.session_state:
            st.write("SHAP Results Loaded")
    
    if st.session_state.shap_status == "error":
        # 展示错误信息
        st.error("⚠️ Error occurred during SHAP analysis")
        
        # 获取详细错误信息
        try:
            error_response = requests.get(f"{API_URL}/api/shap/error")
            if error_response.status_code == 200:
                error_data = error_response.json()
                st.error(f"Error message: {error_data.get('error', 'Unknown error')}")
                
                # 在调试模式下显示更多详情
                if debug_mode and 'details' in error_data and error_data['details']:
                    st.expander("Detailed Error Information", expanded=True).json(error_data['details'])
                    
                # 显示建议
                if 'suggestions' in error_data and error_data['suggestions']:
                    with st.expander("Possible Solutions"):
                        for i, suggestion in enumerate(error_data['suggestions']):
                            st.write(f"{i+1}. {suggestion}")
        except Exception as e:
            st.error(f"Failed to get error details: {str(e)}")
    elif st.session_state.shap_status == "running":
        st.info("⏳ SHAP analysis is in progress...")
        
        # 添加取消按钮（如果分析正在进行）
        if st.button("⚠️ Cancel Ongoing Analysis", key="cancel_shap_in_tab", type="secondary"):
            try:
                requests.post(f"{API_URL}/api/shap/cancel")
                st.success("Cancel request sent")
            except Exception as e:
                st.error(f"Cancel request sending failed: {str(e)}")
                
    elif st.session_state.shap_status == "completed" or st.session_state.show_shap_results:
        # 尝试获取SHAP结果
        has_results = get_shap_results()
        
        st.success("✅ SHAP analysis completed!")
        st.markdown("## SHAP Analysis Results")
        
        if has_results and 'shap_results' in st.session_state:
            results = st.session_state.shap_results
            images = results.get("images", {})
            
            # 使用API提供的图像路径
            st.subheader("Feature Importance Ranking")
            if "bar_plot" in images:
                st.image(f"{API_URL}/{images['bar_plot']}")
            else:
                st.warning("Feature importance chart not found")
                
            st.subheader("Feature Impact Distribution")
            if "beeswarm_plot" in images:
                st.image(f"{API_URL}/{images['beeswarm_plot']}")
            else:
                st.warning("Feature distribution chart not found")
                
            st.subheader("Sample Feature Contribution")
            if "waterfall_plot" in images:
                st.image(f"{API_URL}/{images['waterfall_plot']}")
            else:
                st.warning("Feature contribution chart not found")
                
            # 新增显示summary_plot
            st.subheader("Feature Overall Distribution")
            if "summary_plot" in images:
                st.image(f"{API_URL}/{images['summary_plot']}")
            else:
                st.warning("Feature overall distribution chart not found")
                
            # 新增显示force_plot
            st.subheader("Feature Force Plot")
            if "force_plot" in images:
                st.image(f"{API_URL}/{images['force_plot']}")
            else:
                st.warning("Feature force plot not found")
                
            # 新增显示global_importance
            st.subheader("Global Feature Importance")
            if "global_importance" in images:
                st.image(f"{API_URL}/{images['global_importance']}")
            else:
                st.warning("Global feature importance chart not found")
                
            # 显示特征统计信息
            if "statistics" in results:
                with st.expander("Feature Impact Statistics", expanded=True):
                    stats = results["statistics"]
                    
                    # 创建两列布局
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Positive Impact Features", stats.get("positive_features", "N/A"))
                        st.metric("Max Absolute Importance", f"{stats.get('max_abs_importance', 0):.5f}")
                        
                    with col2:
                        st.metric("Negative Impact Features", stats.get("negative_features", "N/A"))
                        st.metric("Mean Absolute Importance", f"{stats.get('mean_abs_importance', 0):.5f}")
                        
                    # 关键位置信息
                    st.subheader("Key Impact Positions")
                    st.write(f"Drug Key Position: {stats.get('top_drug_position', 'N/A')}")
                    st.write(f"Protein Key Position: {stats.get('top_protein_position', 'N/A')}")
        else:
            # 尝试使用本地文件路径
            st.warning("Unable to get SHAP result images from API, trying local files...")
            
            # 添加SHAP结果图像显示
            st.subheader("Feature Importance Ranking")
            try:
                st.image("static/images/shap_bar_plot.png")
            except:
                try:
                    st.image("shap_bar_plot.png")
                except:
                    st.warning("Cannot load feature importance chart")
                
            st.subheader("Feature Impact Distribution")
            try:
                st.image("static/images/shap_beeswarm_plot.png")
            except:
                try:
                    st.image("shap_beeswarm_plot.png")
                except:
                    st.warning("Cannot load feature distribution chart")
                
            st.subheader("Sample Feature Contribution")
            try:
                st.image("static/images/shap_waterfall_plot.png")
            except:
                try:
                    st.image("shap_waterfall_plot.png")
                except:
                    st.warning("Cannot load feature contribution chart")
                
            # 新增显示summary_plot
            st.subheader("Feature Overall Distribution")
            try:
                st.image("static/images/shap_summary_plot.png")
            except:
                try:
                    st.image("shap_summary_plot.png")
                except:
                    st.warning("Cannot load feature overall distribution chart")
                
            # 新增显示force_plot
            st.subheader("Feature Force Plot")
            try:
                st.image("static/images/shap_force_plot.png")
            except:
                try:
                    st.image("shap_force_plot.png")
                except:
                    st.warning("Cannot load feature force plot")
                
            # 新增显示global_importance
            st.subheader("Global Feature Importance")
            try:
                st.image("static/images/shap_global_importance.png")
            except:
                try:
                    st.image("shap_global_importance.png")
                except:
                    st.warning("Cannot load global feature importance chart")
    else:
        # 未开始分析时的提示
        st.info("Please set parameters in the SHAP Analysis Settings in the sidebar and click 'Start SHAP Analysis'")
        
        # 添加模型检查
        if st.session_state.model_status != "ready":
            st.warning("⚠️ Please load a model before SHAP analysis")
        
        # 添加示例分析按钮
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("View Example Results"):
                # 设置展示演示结果
                st.session_state.show_shap_results = True
                st.experimental_rerun()

# 标签页3: 热力图
with tab3:
    st.subheader("Model Interpretation Heatmap")
    
    # 添加刷新按钮
    refresh_heatmap = st.button("🔄 Refresh Heatmap", key="refresh_heatmap")
    
    # 热力图路径
    heatmap_paths = [
        "static/images/confusion_matrix_heatmap.png",
        "static/images/confusion_matrix_simple.png",
        "static/images/confusion_matrix_example.png",
        "./images/confusion_matrix_heatmap.png",
        "confusion_matrix_heatmap.png"
    ]
    
    if refresh_heatmap or st.session_state.get('show_heatmap', False):
        st.session_state.show_heatmap = True
        
        # 搜索所有可能的热力图路径
        found_heatmap = False
        for path in heatmap_paths:
            if os.path.exists(path):
                st.success(f"✅ Loaded confusion matrix heatmap: {path}")
                
                # 显示文件信息
                file_size = os.path.getsize(path)
                file_time = os.path.getmtime(path)
                file_time_str = datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
                st.info(f"File size: {file_size/1024:.1f} KB, Modified: {file_time_str}")
                
                # 创建空容器显示图像
                heatmap_container = st.empty()
                heatmap_container.image(path)
                
                found_heatmap = True
                break
        
        if not found_heatmap:
            st.warning("Confusion matrix heatmap file not found")
            st.info("Please complete model training first, or ensure the heatmap file has been generated")
            
            # 显示搜索的路径
            with st.expander("View searched file paths"):
                for path in heatmap_paths:
                    exists = "✅" if os.path.exists(path) else "❌"
                    st.text(f"{exists} {path}")

# 标签页4: 训练曲线
with tab4:
    st.subheader("Training Metrics Curves")
    
    # 添加刷新按钮
    refresh_metrics = st.button("🔄 Refresh Training Curves", key="refresh_metrics")
    
    # 训练指标图路径
    metrics_paths = [
        "static/images/training_metrics.png",
        "./images/training_metrics.png",
        "training_metrics.png",
        "static/images/metrics_curve.png" 
    ]
    
    if refresh_metrics or st.session_state.get('show_metrics', False):
        st.session_state.show_metrics = True
        
        # 搜索所有可能的训练曲线路径
        found_metrics = False
        for path in metrics_paths:
            if os.path.exists(path):
                st.success(f"✅ Loaded training curve: {path}")
                
                # 显示文件信息
                file_size = os.path.getsize(path)
                file_time = os.path.getmtime(path)
                file_time_str = datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
                st.info(f"File size: {file_size/1024:.1f} KB, Modified: {file_time_str}")
                
                # 创建空容器显示图像
                metrics_container = st.empty()
                metrics_container.image(path)
                
                found_metrics = True
                break
        
        if not found_metrics:
            st.warning("Training curve file not found")
            
            # 尝试备用方案：从会话状态生成图表
            if st.session_state.training_status == "completed" and st.session_state.training_metrics:
                st.info("Trying to generate curves from training metrics data...")
                try:
                    if "train_loss" in st.session_state.training_metrics and "val_loss" in st.session_state.training_metrics:
                        df = pd.DataFrame({
                            "Training Loss": st.session_state.training_metrics["train_loss"],
                            "Validation Loss": st.session_state.training_metrics["val_loss"]
                        })
                        st.line_chart(df)
                        
                        # 如果有更多指标，也绘制它们
                        metrics = {}
                        if "val_auc" in st.session_state.training_metrics:
                            metrics["Validation AUC"] = st.session_state.training_metrics["val_auc"]
                        if "val_auprc" in st.session_state.training_metrics:
                            metrics["Validation AUPRC"] = st.session_state.training_metrics["val_auprc"]
                        if "val_f1" in st.session_state.training_metrics:
                            metrics["Validation F1"] = st.session_state.training_metrics["val_f1"]
                            
                        if metrics:
                            st.subheader("Validation Metrics")
                            st.line_chart(pd.DataFrame(metrics))
                except Exception as e:
                    st.error(f"Error generating training curves: {str(e)}")
                    st.exception(e)
            else:
                st.info("Please complete model training first, or ensure the training curve file has been generated")
            
            # 显示搜索的路径
            with st.expander("View searched file paths"):
                for path in metrics_paths:
                    exists = "✅" if os.path.exists(path) else "❌"
                    st.text(f"{exists} {path}")
                    
            # 显示静态/图片目录内容
            with st.expander("Static directory contents"):
                for root_dir in ["static/images", "./images", "."]:
                    if os.path.exists(root_dir):
                        st.write(f"**Directory: {root_dir}**")
                        files = os.listdir(root_dir)
                        for f in files:
                            if f.endswith('.png') or f.endswith('.jpg'):
                                full_path = os.path.join(root_dir, f)
                                file_size = os.path.getsize(full_path)
                                st.text(f"{f} ({file_size/1024:.1f} KB)")
                    else:
                        st.text(f"Directory does not exist: {root_dir}")

