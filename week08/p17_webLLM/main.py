"""
LangGraph工作流服务的FastAPI应用程序。
"""
import logging
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from workflow import graph, RequestData
from config import config
from database import db_manager, ConversationHistory


# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="LangGraph工作流API",
    description="使用LangGraph工作流处理用户查询的API",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库。"""
    try:
        # 直接初始化数据库，不使用Celery
        logger.info("开始初始化数据库...")
        db_manager.init_database()
        logger.info("数据库初始化完成")
            
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        logger.warning("应用将继续运行，但数据库功能不可用")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """未处理错误的全局异常处理器。"""
    logger.error(f"未处理的错误: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "内部服务器错误", "detail": str(exc)}
    )


@app.post("/run")
async def run_workflow(data: RequestData) -> Dict[str, Any]:
    """
    通过LangGraph工作流处理用户输入。
    
    Args:
        data: 包含用户输入和会话ID的请求数据
        
    Returns:
        包含工作流结果的字典
        
    Raises:
        HTTPException: 如果工作流执行失败
    """
    try:
        logger.info(f"处理请求，输入长度: {len(data.user_input)}, 会话ID: {data.session_id}")
        result = graph.invoke({
            "user_input": data.user_input,
            "session_id": data.session_id
        })
        
        if "answer" not in result:
            raise HTTPException(
                status_code=500, 
                detail="工作流未返回预期的答案格式"
            )
        
        logger.info("请求处理成功")
        return {
            "success": True,
            "result": result["answer"],
            "session_id": result.get("session_id", data.session_id),
            "input_length": len(data.user_input)
        }
        
    except Exception as e:
        logger.error(f"处理工作流时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"工作流执行失败: {str(e)}"
        )


@app.get("/history")
async def get_conversation_history(
    limit: int = Query(default=50, ge=1, le=200, description="返回记录数限制"),
    session_id: Optional[str] = Query(default=None, description="会话ID过滤")
) -> Dict[str, Any]:
    """
    获取对话历史。
    
    Args:
        limit: 返回记录数限制 (1-200)
        session_id: 会话ID过滤（可选）
        
    Returns:
        包含对话历史的字典
        
    Raises:
        HTTPException: 如果查询失败
    """
    try:
        logger.info(f"查询对话历史，限制: {limit}, 会话ID: {session_id}")
        
        # 直接获取对话历史，不使用Celery
        history = db_manager.get_conversation_history(limit=limit, session_id=session_id)
        
        # 转换为字典格式
        history_list = [
            {
                "id": item.id,
                "user_input": item.user_input,
                "ai_response": item.ai_response,
                "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                "session_id": item.session_id
            }
            for item in history
        ]
        
        logger.info(f"成功获取 {len(history_list)} 条对话历史")
        return {
            "success": True,
            "count": len(history_list),
            "history": history_list
        }
        
    except Exception as e:
        logger.error(f"获取对话历史时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取对话历史失败: {str(e)}"
        )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """健康检查端点。"""
    return {"status": "healthy", "message": "LangGraph工作流API运行正常"}


if __name__ == "__main__":
    logger.info(f"在 {config.HOST}:{config.PORT} 启动服务器")
    uvicorn.run(
        app, 
        host=config.HOST, 
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower()
    )

# 示例curl命令:
# curl -X POST "http://127.0.0.1:8000/run"   -H "Content-Type: application/json"      -d '{"user_input":"什么是 AI 智能体?"}'