"""
LangGraph工作流服务的FastAPI应用程序。
"""
import logging
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from workflow import graph, RequestData
from config import config


# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="LangGraph工作流API",
    description="使用LangGraph工作流处理用户查询的API",
    version="1.0.0"
)


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
        data: 包含用户输入的请求数据
        
    Returns:
        包含工作流结果的字典
        
    Raises:
        HTTPException: 如果工作流执行失败
    """
    try:
        logger.info(f"处理请求，输入长度: {len(data.user_input)}")
        result = graph.invoke({"user_input": data.user_input})
        
        if "answer" not in result:
            raise HTTPException(
                status_code=500, 
                detail="工作流未返回预期的答案格式"
            )
        
        logger.info("请求处理成功")
        return {
            "success": True,
            "result": result["answer"],
            "input_length": len(data.user_input)
        }
        
    except Exception as e:
        logger.error(f"处理工作流时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"工作流执行失败: {str(e)}"
        )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """健康检查端点。"""
    return {"status": "健康", "service": "LangGraph工作流API"}


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