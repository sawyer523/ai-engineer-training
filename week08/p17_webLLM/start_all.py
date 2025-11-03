"""
启动脚本，同时运行FastAPI服务器和Gradio界面。
"""
import logging
import subprocess
import sys
import time
import threading
from pathlib import Path

from config import config

# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


def start_fastapi():
    """启动FastAPI服务器。"""
    try:
        logger.info("启动FastAPI服务器...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", config.HOST, 
            "--port", str(config.PORT),
            "--reload"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FastAPI服务器启动失败: {e}")
    except KeyboardInterrupt:
        logger.info("FastAPI服务器已停止")


def start_celery_worker():
    """启动Celery worker。"""
    try:
        logger.info("启动Celery worker...")
        subprocess.run([sys.executable, "start_celery.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Celery worker启动失败: {e}")
    except KeyboardInterrupt:
        logger.info("Celery worker已停止")


def start_gradio():
    """启动Gradio界面。"""
    try:
        # 等待FastAPI服务器启动
        time.sleep(3)
        logger.info("启动Gradio界面...")
        subprocess.run([sys.executable, "gradio_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Gradio界面启动失败: {e}")
    except KeyboardInterrupt:
        logger.info("Gradio界面已停止")


def main():
    """主函数。"""
    logger.info("开始启动LangGraph AI对话系统...")
    
    # 检查必要文件是否存在
    required_files = ["main.py", "gradio_app.py", "config.py", "database.py", "workflow.py"]
    for file in required_files:
        if not Path(file).exists():
            logger.error(f"缺少必要文件: {file}")
            return
    
    try:
        # 在单独的线程中启动FastAPI
        fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
        fastapi_thread.start()
        
        # 在单独的线程中启动Celery worker
        celery_thread = threading.Thread(target=start_celery_worker, daemon=True)
        celery_thread.start()
        
        # 在主线程中启动Gradio
        start_gradio()
        
    except KeyboardInterrupt:
        logger.info("正在停止所有服务...")
    except Exception as e:
        logger.error(f"启动过程中出错: {e}")


if __name__ == "__main__":
    main()