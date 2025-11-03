"""
Celery任务模块 - 将数据库操作封装为异步任务。
"""
import logging
import json
from typing import List, Optional, Dict, Any
from celery import current_task
from celery.signals import task_prerun, task_postrun, task_failure, task_success
from celery_app import celery_app
from database import db_manager, ConversationHistory
from config import Config
from database_sqlite import sqlite_db_manager

# 根据配置选择数据库管理器
current_db_manager = sqlite_db_manager if Config.DB_TYPE.lower() == "sqlite" else db_manager

logger = logging.getLogger(__name__)

# Celery信号处理器，用于详细日志记录
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """任务开始前的日志记录"""
    logger.info(f"🚀 [CELERY] 任务开始执行 - 任务ID: {task_id}, 任务名称: {task.name}")
    logger.info(f"📝 [CELERY] 任务参数 - args: {args}, kwargs: {kwargs}")
    logger.info(f"🔄 [CELERY] 队列信息 - 发送者: {sender}")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """任务完成后的日志记录"""
    logger.info(f"✅ [CELERY] 任务执行完成 - 任务ID: {task_id}, 状态: {state}")
    logger.info(f"📊 [CELERY] 任务结果 - 返回值: {retval}")

@task_success.connect
def task_success_handler(sender=None, result=None, **kwds):
    """任务成功的日志记录"""
    logger.info(f"🎉 [CELERY] 任务执行成功 - 任务: {sender.name}, 结果: {result}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """任务失败的日志记录"""
    logger.error(f"❌ [CELERY] 任务执行失败 - 任务ID: {task_id}, 任务: {sender.name}")
    logger.error(f"💥 [CELERY] 错误信息 - 异常: {exception}, 堆栈: {traceback}")


@celery_app.task(bind=True, name='celery_tasks.save_conversation_task')
def save_conversation_task(self, user_input: str, ai_response: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    异步保存对话到数据库的任务。
    
    Args:
        user_input: 用户输入
        ai_response: AI响应
        session_id: 会话ID（可选）
        
    Returns:
        包含任务结果的字典
    """
    try:
        logger.info(f"💾 [SAVE_CONVERSATION] 开始保存对话任务，任务ID: {self.request.id}")
        logger.info(f"📄 [SAVE_CONVERSATION] 对话内容 - 用户输入长度: {len(user_input)}, AI响应长度: {len(ai_response)}, 会话ID: {session_id}")
        
        record_id = current_db_manager.save_conversation(user_input, ai_response, session_id)
        
        result = {
            "success": True,
            "record_id": record_id,
            "message": "对话保存成功",
            "task_id": self.request.id
        }
        logger.info(f"✅ [SAVE_CONVERSATION] 对话保存任务完成，记录ID: {record_id}")
        logger.info(f"📊 [SAVE_CONVERSATION] 任务结果: {json.dumps(result, ensure_ascii=False)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ [SAVE_CONVERSATION] 保存对话任务失败: {str(e)}")
        # 重试机制
        if self.request.retries < 3:
            logger.info(f"🔄 [SAVE_CONVERSATION] 重试保存对话任务，重试次数: {self.request.retries + 1}")
            raise self.retry(countdown=60, max_retries=3)
        
        return {
            "success": False,
            "error": str(e),
            "message": "对话保存失败",
            "task_id": self.request.id
        }


@celery_app.task(bind=True, name='celery_tasks.get_conversation_history_task')
def get_conversation_history_task(self, limit: int = 50, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    异步获取对话历史的任务。
    
    Args:
        limit: 返回记录数限制
        session_id: 会话ID过滤（可选）
        
    Returns:
        包含对话历史的字典
    """
    try:
        logger.info(f"📚 [GET_HISTORY] 开始获取对话历史任务，任务ID: {self.request.id}")
        logger.info(f"🔍 [GET_HISTORY] 查询参数 - 限制: {limit}, 会话ID: {session_id}")
        
        conversations = current_db_manager.get_conversation_history(limit=limit, session_id=session_id)
        
        # 转换为字典格式
        history_data = []
        for conv in conversations:
            history_data.append({
                "id": conv.id,
                "user_input": conv.user_input,
                "ai_response": conv.ai_response,
                "timestamp": conv.timestamp.isoformat() if conv.timestamp else None,
                "session_id": conv.session_id
            })
        
        result = {
            "success": True,
            "count": len(history_data),
            "history": history_data,
            "message": "对话历史获取成功",
            "task_id": self.request.id
        }
        logger.info(f"✅ [GET_HISTORY] 对话历史获取任务完成，返回 {len(history_data)} 条记录")
        logger.info(f"📊 [GET_HISTORY] 任务结果: 成功获取 {len(history_data)} 条对话记录")
        return result
        
    except Exception as e:
        logger.error(f"❌ [GET_HISTORY] 获取对话历史任务失败: {str(e)}")
        # 重试机制
        if self.request.retries < 3:
            logger.info(f"🔄 [GET_HISTORY] 重试获取对话历史任务，重试次数: {self.request.retries + 1}")
            raise self.retry(countdown=60, max_retries=3)
        
        return {
            "success": False,
            "error": str(e),
            "message": "对话历史获取失败",
            "task_id": self.request.id
        }


@celery_app.task(bind=True, name='celery_tasks.delete_conversation_history_task')
def delete_conversation_history_task(self, days_old: int = 30) -> Dict[str, Any]:
    """
    异步删除旧对话历史的任务。
    
    Args:
        days_old: 删除多少天前的记录
        
    Returns:
        包含删除结果的字典
    """
    try:
        logger.info(f"🗑️ [DELETE_HISTORY] 开始删除对话历史任务，任务ID: {self.request.id}")
        logger.info(f"📅 [DELETE_HISTORY] 删除参数 - 删除 {days_old} 天前的记录")
        
        deleted_count = current_db_manager.delete_conversation_history(days_old)
        
        result = {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"成功删除 {deleted_count} 条旧对话记录",
            "task_id": self.request.id
        }
        logger.info(f"✅ [DELETE_HISTORY] 对话历史删除任务完成，删除了 {deleted_count} 条记录")
        logger.info(f"📊 [DELETE_HISTORY] 任务结果: {json.dumps(result, ensure_ascii=False)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ [DELETE_HISTORY] 删除对话历史任务失败: {str(e)}")
        # 重试机制
        if self.request.retries < 3:
            logger.info(f"🔄 [DELETE_HISTORY] 重试删除对话历史任务，重试次数: {self.request.retries + 1}")
            raise self.retry(countdown=60, max_retries=3)
        
        return {
            "success": False,
            "error": str(e),
            "message": "对话历史删除失败",
            "task_id": self.request.id
        }


@celery_app.task(bind=True, name='celery_tasks.init_database_task')
def init_database_task(self) -> Dict[str, Any]:
    """
    异步初始化数据库的任务。
    
    Returns:
        包含初始化结果的字典
    """
    try:
        logger.info(f"🔧 [INIT_DB] 开始数据库初始化任务，任务ID: {self.request.id}")
        logger.info(f"🏗️ [INIT_DB] 正在初始化数据库表结构...")
        
        current_db_manager.init_database()
        
        result = {
            "success": True,
            "message": "数据库初始化成功",
            "task_id": self.request.id
        }
        logger.info(f"✅ [INIT_DB] 数据库初始化任务完成")
        logger.info(f"📊 [INIT_DB] 任务结果: {json.dumps(result, ensure_ascii=False)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ [INIT_DB] 数据库初始化任务失败: {str(e)}")
        # 重试机制
        if self.request.retries < 3:
            logger.info(f"🔄 [INIT_DB] 重试数据库初始化任务，重试次数: {self.request.retries + 1}")
            raise self.retry(countdown=60, max_retries=3)
        
        return {
            "success": False,
            "error": str(e),
            "message": "数据库初始化失败",
            "task_id": self.request.id
        }


# 辅助函数：获取任务状态
def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    获取Celery任务的状态。
    
    Args:
        task_id: 任务ID
        
    Returns:
        包含任务状态的字典
    """
    try:
        logger.info(f"🔍 [TASK_STATUS] 查询任务状态，任务ID: {task_id}")
        result = celery_app.AsyncResult(task_id)
        
        status_info = {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "traceback": result.traceback if result.failed() else None
        }
        
        logger.info(f"📊 [TASK_STATUS] 任务状态查询结果: {json.dumps(status_info, ensure_ascii=False)}")
        return status_info
        
    except Exception as e:
        logger.error(f"❌ [TASK_STATUS] 查询任务状态失败: {str(e)}")
        return {
            "task_id": task_id,
            "status": "ERROR",
            "error": str(e)
        }