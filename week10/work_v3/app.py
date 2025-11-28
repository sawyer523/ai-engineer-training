"""FastAPI 应用入口

提供 `/chat` 对话接口与 `/health` 健康检查，支持请求 ID 中间件。
"""
import logging
import uuid
import time
import os
import json
import sqlite3
import hashlib
from functools import wraps
from datetime import datetime
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS as _FAISS

try:
    from . import graph as graph
    from .graph import build_graph, get_react_agent, gen_suggest_questions
    from . import config
    from . import gradio_ui
    from .security_middleware import RedactionMiddleware, build_default_config, sanitize_dict
    from .tools import getdb
except Exception:
    import graph as graph
    from graph import build_graph, get_react_agent, gen_suggest_questions
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
    query: Optional[str] = None
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    images: Optional[list[str]] = None
    audio: Optional[str] = None
    asr_language: Optional[str] = None
    asr_itn: Optional[bool] = None


app = FastAPI()
app.add_middleware(RedactionMiddleware, config=build_default_config())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_chain = build_graph()
gradio_ui.mount_gradio(app)
SUGGEST_QUEUES = {}
import asyncio
from fastapi.responses import StreamingResponse
try:
    from .mcp_server import mcp as _mcp
except Exception:
    from mcp_server import mcp as _mcp
try:
    _mcp_app = _mcp.sse_app()
    app.mount("/mcp", _mcp_app)
except Exception:
    pass

VECTORS_LOCK = asyncio.Lock()

class SwitchRequest(BaseModel):
    name: str

def _ok(data: Any) -> Dict[str, Any]:
    return {"code": 0, "message": "OK", "data": data}

def _err(code: str, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"code": code, "message": message, "data": data or {}}

def _stable_id_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def require_api_key(request: Request):
    key = "test"
    if not key or request.headers.get("X-API-Key") != key:
        raise HTTPException(status_code=401, detail="Unauthorized")

def _audit(op: str, data: Dict[str, Any]):
    try:
        base = os.path.dirname(__file__)
        logs = os.path.join(base, "logs")
        os.makedirs(logs, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        line = f"{ts} | INFO | audit | {op} " + json.dumps(sanitize_dict(data), ensure_ascii=False)
        with open(os.path.join(logs, "requests.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

class VectorItem(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

class VectorsAddRequest(BaseModel):
    items: List[VectorItem]

class VectorsDeleteRequest(BaseModel):
    ids: List[str]


# 计时与指标装饰器：异步统计耗时并做日志与分类更新（保持与原逻辑一致）
def measure_latency(func):
    @wraps(func)
    async def _wrap(*args, **kwargs):
        # 开始计时
        _start = time.perf_counter()
        # 执行原始处理函数
        result = await func(*args, **kwargs)
        # 仅在返回包含 route 的结果时统计与写日志（命令分支不统计）
        if isinstance(result, dict) and "route" in result:
            _elapsed_ms = (time.perf_counter() - _start) * 1000.0
            _route = result.get("route")
            m = config.get_metrics()
            # 更新总体耗时
            m.update("overall", _elapsed_ms)
            # 根据路由分类到对应指标维度
            _cat = "direct"
            if _route in {"course", "presale", "postsale"}:
                _cat = "kb"
            elif _route == "order":
                _cat = "order"
            elif _route == "human":
                _cat = "handoff"
            m.update(_cat, _elapsed_ms)
            logging.info("latency route=%s cost=%.2fms", _route, _elapsed_ms)
            try:
                # 终端日志行（与原格式一致）
                _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                _term_line = f"{_ts} | INFO | root | latency route={_route} cost={_elapsed_ms:.2f}ms"
                # 对请求做脱敏，仅保留必要字段，避免泄露敏感信息
                _req = args[0] if len(args) > 0 else kwargs.get("req")
                _safe = sanitize_dict({"query": getattr(_req, "query", None) if _req else None})
                _req_str = json.dumps(_safe, ensure_ascii=False)
                # 写入本地日志文件（requests.log）
                _base = os.path.dirname(__file__)
                _logs = os.path.join(_base, "logs")
                os.makedirs(_logs, exist_ok=True)
                with open(os.path.join(_logs, "requests.log"), "a", encoding="utf-8") as f:
                    f.write(_term_line + " | request=" + _req_str + "\n")
            except Exception:
                pass
        return result
    return _wrap

# 处理 /help /history /reset 指令；未匹配返回 None
def _handle_command(query_text: str, thread_id: str):
    if query_text.strip().startswith("/"):
        cmd = query_text.strip().lower()
        if cmd.startswith("/help"):
            return {
                "commands": [
                    {"cmd": "/help", "desc": "查看所有快捷指令"},
                    {"cmd": "/history", "desc": "查看最近5轮对话"},
                    {"cmd": "/reset", "desc": "重置当前会话上下文"},
                ]
            }
        if cmd.startswith("/history"):
            msgs = config.get_session_messages(thread_id, maxlen=5)
            return {"history": msgs}
        if cmd.startswith("/reset"):
            config.reset_session(thread_id)
            return {"reset": True}
    return None

def _determine_answer(result: Dict[str, Any]):
    route = result.get("route") or result.get("intent")
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
    return route, answer, sources

async def _push_suggest(thread_id: str, query: Optional[str], answer: str, route: Optional[str]):
    # 异步推送建议：先发送 react_start，再尝试生成建议，失败则发送 error
    try:
        q = SUGGEST_QUEUES.get(thread_id)
        if q is None:
            q = asyncio.Queue()
            SUGGEST_QUEUES[thread_id] = q
        await asyncio.sleep(0.05)
        await q.put({"route": route, "suggestions": [], "event": "react_start"})
        try:
            suggestions = await gen_suggest_questions(thread_id, query, answer, route)
            await q.put({"route": route, "suggestions": suggestions, "event": "react", "final": True})
        except Exception:
            await q.put({"route": route, "error": {"code": "react_error", "message": "建议生成异常"}, "event": "error", "final": True})
    except Exception:
        pass

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """为每个请求注入 `X-Request-Id`，便于链路追踪。"""
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    # 将生成的请求 ID 绑定到 request.state，方便后续业务逻辑获取链路追踪标识
    request.state.request_id = request_id
    # 将请求 ID 绑定到日志记录上下文，方便日志跟踪
    logging.info(f"Request {request_id} received: {request.method} {request.url}")
    # 将请求继续传递给下游的 FastAPI 路由或中间件，并等待其处理完成
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.post("/chat")
@measure_latency
async def chat(req: ChatRequest, request: Request):
    """对话接口：构建状态并调用图执行，返回答案与来源。"""
    # 生成或复用会话唯一标识
    thread_id = req.thread_id or request.state.request_id
    # 取用户输入并去掉首尾空白
    audio_text = graph.transcribe_audio(req.audio, req.asr_language, True if req.asr_itn is None else bool(req.asr_itn)) if getattr(req, "audio", None) else None
    query_text = (req.query or audio_text or "").strip()
    # 指令模式优先处理并返回
    cmd = _handle_command(query_text, thread_id)
    if cmd is not None:
        return cmd
    # 取最近 5 条历史消息，用于 LLM 上下文
    session_msgs = config.get_session_messages(thread_id, maxlen=5)
    history = []
    # 格式化历史为 "role: content" 的行
    history.extend(f"{m.get('role')}: {m.get('content')}" for m in session_msgs)
    tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
    state: Dict[str, Any] = {"query": query_text, "history": "\n".join(history), "tenant_id": tenant_id}
    cfg = {"configurable": {"thread_id": thread_id}}
    if query_text:
        config.append_session_message(thread_id, "user", query_text)
    need_vl = bool(getattr(req, "images", None))
    t = query_text.lower()
    is_kb = any(k in t for k in ["课程", "售前", "售后", "新手"])
    is_order = any(k in t for k in ["订单", "支付", "退款", "order", "status"])
    use_vl = need_vl and (is_kb or is_order)
    prev_llm = getattr(graph, "llm", None)
    prev_router = getattr(graph, "router_llm", None)
    prev_sql = getattr(graph, "sql_llm", None)
    if use_vl:
        try:
            from langchain_community.chat_models import ChatTongyi as _CT
            graph.llm = _CT(model="qwen-vl-max")
            graph.router_llm = graph.llm.with_structured_output(graph.Route)
            graph.sql_llm = graph.llm.with_structured_output(graph.SQLSpec)
        except Exception:
            pass
    try:
        chain = build_graph(tenant_id)
        result = await chain.ainvoke(state, cfg)
    finally:
        if use_vl:
            graph.llm = prev_llm
            graph.router_llm = prev_router
            graph.sql_llm = prev_sql
    # 依据结果选择答案与来源
    route, answer, sources = _determine_answer(result)
    # 把助手回复写入会话历史，便于后续上下文构造
    if answer:
        config.append_session_message(thread_id, "assistant", answer)
    # 异步启动建议问题推送，不阻塞主流程
    if asyncio is not None:
        asyncio.create_task(_push_suggest(thread_id, query_text, answer, route))
    # 返回路由、答案与来源
    return {"route": route, "answer": answer, "sources": sources}


@app.get("/models/list")
async def models_list():
    t0 = time.perf_counter()
    items = config.get_supported_models()
    cur = config.get_current_model_name()
    data = {"current": cur, "models": items}
    _elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logging.info("models list cost=%.2fms", _elapsed_ms)
    return _ok(data)


@app.post("/api/v1/vectors/items")
async def vectors_add(req: VectorsAddRequest, request: Request, _auth: Any = Depends(require_api_key)):
    t0 = time.perf_counter()
    tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
    vs = config.get_vector_store(tenant_id)
    async with VECTORS_LOCK:
        ds = getattr(vs, "docstore", None)
        d = getattr(ds, "_dict", None) if ds else None
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        skipped: List[str] = []
        for item in req.items:
            tid = item.id or _stable_id_text(item.text)
            if d is not None and tid in d:
                skipped.append(tid)
                continue
            m = dict(item.metadata) if isinstance(item.metadata, dict) else {"source": "api"}
            m["id"] = tid
            texts.append(item.text)
            metadatas.append(sanitize_dict(m))
            ids.append(tid)
        added = 0
        if ids:
            try:
                if vs is None:
                    emb = config.get_embeddings()
                    vs = _FAISS.from_texts(texts, embedding=emb, metadatas=metadatas, ids=ids)
                    try:
                        import tempfile, shutil, os as _os
                        ascii_dir = _os.path.join(tempfile.gettempdir(), "kb_index_add_" + tenant_id)
                        _os.makedirs(ascii_dir, exist_ok=True)
                        vs.save_local(ascii_dir)
                        target = config.get_kb_index_dir(tenant_id)
                        _os.makedirs(target, exist_ok=True)
                        for name in ("index.faiss", "index.pkl"):
                            src = _os.path.join(ascii_dir, name)
                            dst = _os.path.join(target, name)
                            if _os.path.isfile(src):
                                shutil.copyfile(src, dst)
                    except Exception:
                        pass
                else:
                    vs.add_texts(texts, metadatas=metadatas, ids=ids)
                added = len(ids)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    elapsed = (time.perf_counter() - t0) * 1000.0
    config.get_metrics().update("vectors_add", elapsed)
    _audit("vectors_add", {"request_id": request.state.request_id, "added": added, "skipped": len(skipped), "ids": ids[:5]})
    return _ok({"added": added, "ids": ids, "skipped": skipped})


@app.delete("/api/v1/vectors/items")
async def vectors_delete(req: VectorsDeleteRequest, request: Request, _auth: Any = Depends(require_api_key)):
    t0 = time.perf_counter()
    tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
    vs = config.get_vector_store(tenant_id)
    if vs is None:
        raise HTTPException(status_code=500, detail="Vector store unavailable")
    async with VECTORS_LOCK:
        ds = getattr(vs, "docstore", None)
        d = getattr(ds, "_dict", None) if ds else None
        affected = 0
        try:
            if hasattr(vs, "delete"):
                vs.delete(ids=req.ids)
                if d is not None:
                    affected = sum(1 for i in req.ids if i in d)
                else:
                    affected = len(req.ids)
            else:
                raise HTTPException(status_code=400, detail="Delete not supported")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elapsed = (time.perf_counter() - t0) * 1000.0
    config.get_metrics().update("vectors_delete", elapsed)
    _audit("vectors_delete", {"request_id": request.state.request_id, "deleted": affected, "ids": req.ids[:5]})
    return _ok({"deleted": affected, "ids": req.ids})


@app.post("/models/switch")
async def models_switch(req: SwitchRequest):
    name = str(req.name or "").strip()
    tenant_id = None
    lock = config.get_model_lock()
    with lock:
        v = config.validate_model(name)
        if not v.get("ok"):
            return _err(v.get("code") or "invalid", v.get("message") or "无效模型")
        res = config.switch_model(name, tenant_id)
        if not res.get("ok"):
            return _err(res.get("code") or "error", res.get("message") or "切换失败")
        try:
            graph.llm = config.get_llm()
            graph.router_llm = graph.llm.with_structured_output(graph.Route)
            graph.sql_llm = graph.llm.with_structured_output(graph.SQLSpec)
            graph._react_agent = None
            global _chain
            _chain = build_graph()
        except Exception as e:
            return _err("graph_reload_error", str(e))
        try:
            base = os.path.dirname(__file__)
            logs = os.path.join(base, "logs")
            os.makedirs(logs, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            line = f"{ts} | INFO | root | model switch apply new={name}"
            with open(os.path.join(logs, "requests.log"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        return _ok({"current": config.get_current_model_name(), "models": config.get_supported_models()})


@app.get("/health")
async def health():
    """健康检查：返回模型名、KB 索引是否可用、订单库路径是否存在。"""
    vs_ok = config.get_vector_store() is not None
    db_order = bool(config.get_orders_db_path())
    m = config.get_metrics()
    # 汇总各类路由的耗时指标快照
    metrics = {
        "overall": m.snapshot("overall"),
        "kb": m.snapshot("kb"),
        "order": m.snapshot("order"),
        "direct": m.snapshot("direct"),
        "handoff": m.snapshot("handoff"),
        "vectors_add": m.snapshot("vectors_add"),
        "vectors_delete": m.snapshot("vectors_delete"),
    }
    return {"model": config.MODEL_NAME, "kb_index": vs_ok, "orders_db": db_order, "metrics": metrics}


@app.get("/greet")
async def greet():
    # 返回欢迎语与常用入口，便于前端引导
    return {
        "message": "您好，请问有什么可以帮您？",
        "options": [
            {"key": "course", "title": "课程咨询", "desc": "显示课程目录和详细信息"},
            {"key": "order", "title": "订单查询", "desc": "验证用户身份后显示订单状态"},
            {"key": "human", "title": "人工转接", "desc": "直接转人工客服"},
        ],
    }


@app.get("/suggest/{thread_id}")
async def suggest(thread_id: str):
    async def _gen():
        q = SUGGEST_QUEUES.setdefault(thread_id, asyncio.Queue())
        deadline = time.perf_counter() + 15.0
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if time.perf_counter() > deadline:
                    err = {"route": None, "error": {"code": "timeout", "message": "建议生成超时"}, "event": "error", "final": True}
                    data = json.dumps(err, ensure_ascii=False)
                    yield f"id: {thread_id}\nevent: error\ndata: {data}\n\n"
                    break
                continue
            data = json.dumps(item, ensure_ascii=False)
            ev = str(item.get("event", "suggest"))
            yield f"id: {thread_id}\nevent: {ev}\ndata: {data}\n\n"
            if bool(item.get("final")):
                break
    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/api/orders/{order_id}")
async def get_order(order_id: str, request: Request):
    try:
        payload = getdb(order_id)
        sql, params = payload["sql"], payload["params"]
        tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
        db_path = config.get_orders_db_path(tenant_id)
        if not db_path:
            raise HTTPException(status_code=500, detail="Orders database not configured")
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(sql.replace("%s", "?"), params).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Order not found")
        return {
            "order_id": str(row[0]),
            "status": str(row[1]),
            "amount": float(row[2]) if row[2] is not None else None,
            "updated_at": str(row[3]) if row[3] is not None else None,
            "enroll_time": None,
            "start_time": str(row[4]) if len(row) > 4 and row[4] is not None else None,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from langchain_community.vectorstores import FAISS as _FAISS
