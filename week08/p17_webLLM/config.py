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
    
    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/langraph_db")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "langraph_db")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "password")
    
    # 数据库类型选择
    DB_TYPE: str = os.getenv("DB_TYPE", "sqlite")  # sqlite or postgresql
    
    # Redis配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    @classmethod
    def get_tongyi_api_key(cls) -> Optional[str]:
        """从环境变量获取通义API密钥。"""
        return os.getenv("DASHSCOPE_API_KEY")
    
    @classmethod
    def get_database_url(cls) -> str:
        """获取数据库连接URL。"""
        if cls.DB_TYPE.lower() == "sqlite":
            return os.getenv("SQLITE_DB_PATH", "conversation_history.db")
        else:
            return cls.DATABASE_URL or f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"


# 创建全局配置实例
config = Config()