import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .services import service_manager
from .graph import GraphManager
from langchain_core.messages import HumanMessage, AIMessage


# 初始化 FastAPI 应用
app = FastAPI(
    title="Smart Customer Service API",
    description="一个集成了LangGraph和工具热更新的智能客服API",
    version="1.0.0",
)


# 初始化图管理器
# GraphManager 依赖于 ServiceManager
graph_manager = GraphManager(service_manager)


class ChatRequest(BaseModel):
    user_id: str  # 用于追踪会话
    query: str


class HotUpdateRequest(BaseModel):
    type: str  # "model" or "tools"
    name: str  # e.g., "qwen-max" or "default"


@app.get("/health", summary="健康检查")
async def health_check():
    """检查服务是否健康运行"""
    return {"status": "healthy", "services": service_manager.get_services_status()}


@app.post("/chat", summary="进行对话")
async def chat(request: ChatRequest):
    """与智能客服进行单轮对话"""
    thread_id = request.user_id
    config = {"configurable": {"thread_id": thread_id}}

    messages = [HumanMessage(content=request.query)]

    # 使用当前的 graph 实例
    current_app = graph_manager.get_app()

    final_response = ""
    # 流式处理以获取最终回复
    for event in current_app.stream({"messages": messages}, config=config, stream_mode="values"):
        if "messages" in event:
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                final_response = last_message.content

    if not final_response:
        return {"user_id": thread_id, "response": "抱歉，我暂时无法回答这个问题。"}

    return {"user_id": thread_id, "response": final_response}


@app.post("/hot-update", summary="热更新模型或工具")
async def hot_update(request: HotUpdateRequest):
    """
    执行模型或工具的热更新。
    - type: 'model', name: 'qwen-max'
    - type: 'tools', name: 'query_only' (示例)
    """
    try:
        if request.type == "model":
            service_manager.update_llm(request.name)
        elif request.type == "tools":
            # 这里可以根据 name 加载不同的工具集
            if request.name == "query_only":
                from .tools.order_tools import query_order
                service_manager.update_tools([query_order])
            else:  # 恢复默认
                from .tools import default_tools
                service_manager.update_tools(default_tools)
        else:
            raise HTTPException(status_code=400, detail="无效的更新类型")

        # 更新后，重新加载图
        graph_manager.reload_graph()

        return {"status": "success", "message": f"{request.type} 热更新完成."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"热更新失败: {e}")


# 如果你想通过 python -m smart_customer_service.api 运行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
