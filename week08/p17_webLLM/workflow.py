"""
用于处理用户查询的LangGraph工作流模块。
"""
import logging
import uuid
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, constr
from tenacity import retry, wait_exponential, stop_after_attempt

from config import config
from config import Config
from database import db_manager
from database_sqlite import sqlite_db_manager
from celery_tasks import save_conversation_task

# 根据配置选择数据库管理器
current_db_manager = sqlite_db_manager if Config.DB_TYPE.lower() == "sqlite" else db_manager

# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# 初始化LLM
llm = ChatTongyi(model=config.LLM_MODEL, dashscope_api_key=config.get_tongyi_api_key())


class RequestData(BaseModel):
    """API端点的请求数据模型。"""
    user_input: constr(min_length=config.MIN_INPUT_LENGTH, max_length=config.MAX_INPUT_LENGTH)
    session_id: str = None
    
    def __init__(self, **data):
        if 'session_id' not in data or not data['session_id']:
            data['session_id'] = str(uuid.uuid4())
        super().__init__(**data)


@retry(wait=wait_exponential(multiplier=config.RETRY_MULTIPLIER, min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT), 
       stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS))
def safe_invoke_llm(message: str) -> Any:
    """
    使用重试机制安全调用LLM。
    
    Args:
        message: LLM的输入消息
        
    Returns:
        LLM响应对象
    """
    return llm.invoke([HumanMessage(content=message)])


def answer_question(state: Dict[str, Any]) -> Dict[str, str]:
    """
    处理用户输入并使用LLM生成答案。
    
    Args:
        state: 包含user_input和session_id的字典
        
    Returns:
        包含答案或错误消息的字典
    """
    user_input = state["user_input"]
    session_id = state.get("session_id")
    logger.info(f"收到输入: {user_input}, 会话ID: {session_id}")
    
    try:
        response = safe_invoke_llm(user_input)
        ai_response = response.content
        logger.info("LLM响应生成成功")
        
        # 保存对话到数据库（使用Celery异步任务）
        try:
            task = save_conversation_task.delay(user_input, ai_response, session_id)
            logger.info(f"对话保存任务已提交，任务ID: {task.id}")
        except Exception as db_error:
            logger.error(f"提交对话保存任务失败: {str(db_error)}")
            # 即使任务提交失败，也返回AI响应
        
        return {"answer": ai_response, "session_id": session_id}
    except Exception as e:
        logger.error(f"生成响应时出错: {str(e)}")
        error_message = f"错误: {str(e)}"
        
        # 尝试保存错误信息到数据库（使用Celery异步任务）
        try:
            task = save_conversation_task.delay(user_input, error_message, session_id)
            logger.info(f"错误信息保存任务已提交，任务ID: {task.id}")
        except Exception as db_error:
            logger.error(f"提交错误信息保存任务失败: {str(db_error)}")
        
        return {"answer": error_message, "session_id": session_id}


def create_workflow() -> StateGraph:
    """
    创建并配置LangGraph工作流。
    
    Returns:
        编译后的StateGraph工作流
    """
    workflow = StateGraph(dict)
    workflow.add_node("answer", answer_question)
    workflow.add_edge(START, "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()


# 创建图实例
graph = create_workflow()