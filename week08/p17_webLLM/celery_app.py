"""
Celery应用配置模块。
"""
import os
from celery import Celery
from config import config

# 创建Celery应用实例
celery_app = Celery(
    'p17_webLLM',
    broker=f'redis://{config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}',
    backend=f'redis://{config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}',
)

# Celery配置
celery_app.conf.update(
    # 任务序列化
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    
    # 任务结果过期时间（秒）
    result_expires=3600,
    
    # 任务重试配置
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    
    # 任务超时配置
    task_soft_time_limit=300,  # 5分钟软超时
    task_time_limit=600,       # 10分钟硬超时
)

# 自动发现任务
celery_app.autodiscover_tasks()

# 导入任务模块以确保任务被注册
import celery_tasks

if __name__ == '__main__':
    celery_app.start()