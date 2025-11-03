"""
用于处理用户查询的LangGraph工作流模块。
"""
import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, constr
from tenacity import retry, wait_exponential, stop_after_attempt

from config import config

# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# 初始化LLM
llm = ChatTongyi(model=config.LLM_MODEL, dashscope_api_key=config.get_tongyi_api_key())


class RequestData(BaseModel):
    """API端点的请求数据模型。"""
    user_input: constr(min_length=config.MIN_INPUT_LENGTH, max_length=config.MAX_INPUT_LENGTH)


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
        state: 包含user_input的字典
        
    Returns:
        包含答案或错误消息的字典
    """
    user_input = state["user_input"]
    logger.info(f"收到输入: {user_input}")
    
    try:
        response = safe_invoke_llm(user_input)
        logger.info("LLM响应生成成功")
        return {"answer": response.content}
    except Exception as e:
        logger.error(f"生成响应时出错: {str(e)}")
        return {"answer": f"错误: {str(e)}"}


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