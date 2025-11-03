"""
应用程序配置模块。
"""
import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """应用程序配置类。"""
    
    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # LLM配置
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-turbo")
    
    # API配置
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "500"))
    MIN_INPUT_LENGTH: int = int(os.getenv("MIN_INPUT_LENGTH", "1"))
    
    # 重试配置
    RETRY_MULTIPLIER: int = int(os.getenv("RETRY_MULTIPLIER", "1"))
    RETRY_MIN_WAIT: int = int(os.getenv("RETRY_MIN_WAIT", "2"))
    RETRY_MAX_WAIT: int = int(os.getenv("RETRY_MAX_WAIT", "10"))
    RETRY_MAX_ATTEMPTS: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_tongyi_api_key(cls) -> Optional[str]:
        """从环境变量获取通义API密钥。"""
        return os.getenv("DASHSCOPE_API_KEY")


# 创建全局配置实例
config = Config()