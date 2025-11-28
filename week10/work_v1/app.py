"""FastAPI 应用入口

提供 `/chat` 对话接口与 `/health` 健康检查，支持请求 ID 中间件。
"""
import logging
import uuid
import time
import os
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

try:
    from .graph import build_graph
    from . import config
    from . import gradio_ui
    from .security_middleware import RedactionMiddleware, build_default_config, sanitize_dict
    from .tools import getdb
except Exception:
    from graph import build_graph
    import config as config
    import gradio_ui as gradio_ui
    from security_middleware import RedactionMiddleware, build_default_config, sanitize_dict
    from tools import getdb


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


class ChatRequest(BaseModel):
    """聊天请求体

    - query：用户问题
    - user_id：可选的用户标识（用于审计/记录）
    - thread_id：可选的会话线程 ID（用于 LangGraph 检查点）
    """
    query: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None


app = FastAPI()
app.add_middleware(RedactionMiddleware, config=build_default_config())
_chain = build_graph()
gradio_ui.mount_gradio(app)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """为每个请求注入 `X-Request-Id`，便于链路追踪。"""
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """对话接口：构建状态并调用图执行，返回答案与来源。"""
    start = time.perf_counter()
    thread_id = req.thread_id or request.state.request_id
    state: Dict[str, Any] = {"query": req.query}
    cfg = {"configurable": {"thread_id": thread_id}}
    result = await _chain.ainvoke(state, cfg)
    route = result.get("route") or result.get("intent")
    answer = None
    sources = result.get("sources")
    order = result.get("order_summary")
    human = result.get("human_handoff")
    kb = result.get("kb_answer")
    if order:
        answer = order
    elif human:
        answer = human
    elif kb:
        answer = kb
    else:
        answer = ""
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    m = config.get_metrics()
    m.update("overall", elapsed_ms)
    cat = "direct"
    if route in {"course", "presale", "postsale"}:
        cat = "kb"
    elif route == "order":
        cat = "order"
    elif route == "human":
        cat = "handoff"
    m.update(cat, elapsed_ms)
    logging.info("latency route=%s cost=%.2fms", route, elapsed_ms)
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        term_line = f"{ts} | INFO | root | latency route={route} cost={elapsed_ms:.2f}ms"
        safe_req = sanitize_dict({"query": req.query})
        req_str = json.dumps(safe_req, ensure_ascii=False)
        base = os.path.dirname(__file__)
        logs_dir = os.path.join(base, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, "requests.log"), "a", encoding="utf-8") as f:
            f.write(term_line + " | request=" + req_str + "\n")
    except Exception:
        pass
    return {"route": route, "answer": answer, "sources": sources}


@app.get("/health")
async def health():
    """健康检查：返回模型名、KB 索引是否可用、订单库路径是否存在。"""
    vs_ok = config.get_vector_store() is not None
    db_order = bool(config.get_orders_db_path())
    m = config.get_metrics()
    metrics = {
        "overall": m.snapshot("overall"),
        "kb": m.snapshot("kb"),
        "order": m.snapshot("order"),
        "direct": m.snapshot("direct"),
        "handoff": m.snapshot("handoff"),
    }
    return {"model": config.MODEL_NAME, "kb_index": vs_ok, "orders_db": db_order, "metrics": metrics}


@app.get("/api/orders/{order_id}")
async def get_order(order_id: str):
    try:
        payload = getdb(order_id)
        sql = payload.get("sql")
        params = payload.get("params")
        db_path = config.get_orders_db_path()
        if not db_path:
            raise HTTPException(status_code=500, detail="Orders database not configured")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql.replace("%s", "?"), params)
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Order not found")
        data = {
            "order_id": str(row[0]),
            "status": str(row[1]),
            "amount": float(row[2]) if row[2] is not None else None,
            "updated_at": str(row[3]) if row[3] is not None else None,
            "enroll_time": None,
            "start_time": str(row[4]) if len(row) > 4 and row[4] is not None else None,
        }
        return data
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
