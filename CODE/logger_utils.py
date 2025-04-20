import os
import logging
import sys
from datetime import datetime
import inspect
import functools

# 日志级别
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

# 日志格式
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 创建统一的日志配置
def setup_logger(logger_name, log_level='info', log_to_console=True, log_to_file=True):
    """
    配置并返回一个日志记录器
    
    参数:
        logger_name (str): 日志记录器名称
        log_level (str): 日志级别，可选值：debug, info, warning, error, critical
        log_to_console (bool): 是否输出到控制台
        log_to_file (bool): 是否输出到文件
        
    返回:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 获取日志记录器
    logger = logging.getLogger(logger_name)
    
    # 设置日志级别
    logger.setLevel(LOG_LEVELS.get(log_level.lower(), logging.INFO))
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    # 添加文件处理器
    if log_to_file:
        log_filename = f"logs/moltrans_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 添加控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logger.level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.debug(f"日志记录器 '{logger_name}' 初始化完成，级别: {log_level}")
    
    return logger

# 获取预配置的日志记录器
def get_logger(component_name, log_level='info'):
    """
    获取指定组件的日志记录器
    
    参数:
        component_name (str): 组件名称
        log_level (str): 日志级别
        
    返回:
        logging.Logger: 日志记录器实例
    """
    return setup_logger(f"MolTrans.{component_name}", log_level)

# 包级日志记录器
logger = get_logger('Core')

def log_execution_time(func):
    """
    装饰器：记录函数执行时间，支持同步和异步函数
    
    参数:
        func: 要装饰的函数
        
    返回:
        函数的包装器
    """
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # 获取函数所属类的名称（如果是方法）
        if len(args) > 0 and hasattr(args[0], '__class__'):
            component_name = args[0].__class__.__name__
        else:
            component_name = 'Function'
        
        # 获取日志记录器
        func_logger = get_logger(component_name)
        
        func_logger.debug(f"开始执行 {func.__name__}")
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            func_logger.debug(f"{func.__name__} 执行完成，耗时: {execution_time:.2f}秒")
            
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            func_logger.error(f"{func.__name__} 执行出错，耗时: {execution_time:.2f}秒, 错误: {str(e)}")
            raise
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # 获取函数所属类的名称（如果是方法）
        if len(args) > 0 and hasattr(args[0], '__class__'):
            component_name = args[0].__class__.__name__
        else:
            component_name = 'Function'
        
        # 获取日志记录器
        func_logger = get_logger(component_name)
        
        func_logger.debug(f"开始执行异步函数 {func.__name__}")
        start_time = datetime.now()
        
        try:
            result = await func(*args, **kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            func_logger.debug(f"异步函数 {func.__name__} 执行完成，耗时: {execution_time:.2f}秒")
            
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            func_logger.error(f"异步函数 {func.__name__} 执行出错，耗时: {execution_time:.2f}秒, 错误: {str(e)}")
            raise
    
    # 检查函数是否为异步函数
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# 测试日志系统
if __name__ == "__main__":
    test_logger = get_logger("Test")
    test_logger.debug("这是一条调试消息")
    test_logger.info("这是一条信息消息")
    test_logger.warning("这是一条警告消息")
    test_logger.error("这是一条错误消息")
    
    # 测试装饰器
    @log_execution_time
    def test_function():
        import time
        test_logger.info("函数正在执行...")
        time.sleep(1)
        return "测试完成"
    
    # 测试异步装饰器
    @log_execution_time
    async def test_async_function():
        import asyncio
        test_logger.info("异步函数正在执行...")
        await asyncio.sleep(1)
        return "异步测试完成"
    
    # 同步测试
    result = test_function()
    test_logger.info(f"结果: {result}")
    
    # 异步测试需要在异步环境中运行
    # import asyncio
    # async_result = asyncio.run(test_async_function())
    # test_logger.info(f"异步结果: {async_result}") 