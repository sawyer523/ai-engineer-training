"""对话路由与节点执行图

使用 LangGraph 构建状态机：
- intent 节点：根据关键词或 LLM 判断用户意图
- kb 节点：检索知识库并回答
- order 节点：查询订单并生成客服话术
- handoff 节点：未命中/转人工时输出渠道
- direct 节点：无 KB/订单时的直接回答
"""
import re
from typing import Any, Dict, Optional
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

try:
    from .statee import State
    from .prompts import INTENT_PROMPT, RAG_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE, ORDER_NLG_PROMPT_TEMPLATE, ORDER_SQL_PROMPT_TEMPLATE
    from . import config
    from .tools import (
        retrieve_kb,
        getdb,
        exec_sql,
        record_unanswered,
        _format_order_nlg,
        handoff_to_human,
    )
except Exception:
    from statee import State
    from prompts import INTENT_PROMPT, RAG_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE, ORDER_NLG_PROMPT_TEMPLATE, ORDER_SQL_PROMPT_TEMPLATE
    import config as config
    from tools import (
        retrieve_kb,
        getdb,
        exec_sql,
        record_unanswered,
        _format_order_nlg,
        handoff_to_human,
    )


class Route(BaseModel):
    """结构化路由输出模型，用于约束 LLM 返回的意图标签。"""
    step: Literal["course", "presale", "postsale", "order", "human", "direct"] = Field(None)


llm = config.get_llm()
router_llm = llm.with_structured_output(Route)
class SQLSpec(BaseModel):
    sql: Optional[str] = None
    params: Optional[list] = None
sql_llm = llm.with_structured_output(SQLSpec)


def _clean_input(text: str) -> str:
    """清理输入文本：去除多余空白与不可见字符。"""
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\u200b", "")
    return s


def _keywords_intent(text: str) -> Optional[str]:
    """基于关键词的快速意图判断，优先覆盖明显场景。注意不同场景也会有并行判断。"""
    t = (text or "").lower()
    if any(k in t for k in ["人工", "客服", "转人工"]):
        return "human"
    if any(k in t for k in ["订单", "支付", "退款", "order", "status"]):
        return "order"
    if any(k in t for k in ["课程", "售前", "售后", "新手"]):
        return "course"
    return None


def intent_node(state: State) -> Dict[str, Any]:
    """意图识别节点：优先关键词，其次调用 LLM 结构化路由。"""
    q = _clean_input(state.get("query", ""))
    kw = _keywords_intent(q)
    if kw:
        return {"intent": kw, "route": kw}
    try:
        r = router_llm.invoke(INTENT_PROMPT + "\n用户查询：" + q)
        step = getattr(r, "step", None)
        if step in {"course", "presale", "postsale", "order", "human", "direct"}:
            return {"intent": step, "route": step}
    except Exception:
        pass
    return {"intent": "direct", "route": "direct"}


def kb_node(state: State) -> Dict[str, Any]:
    """知识库节点：RAG 检索并依据参考资料作答。"""
    q = state.get("query", "")
    serialized, docs = retrieve_kb(q)
    sources = []
    try:
        sources = [getattr(d, "metadata", {}) for d in docs]
    except Exception:
        sources = []
    if not docs:
        return {"kb_answer": "", "sources": sources}
    prompt = RAG_PROMPT_TEMPLATE.format(context=serialized, question=q)
    msg = llm.invoke(prompt)
    content = str(getattr(msg, "content", msg)).strip()
    return {"kb_answer": content, "sources": sources}


def no_kb_then_handoff_node(state: State) -> Dict[str, Any]:
    """兜底节点：记录未命中问题，并返回转人工渠道信息。"""
    q = state.get("query", "")
    record_unanswered(q)
    payload = {"query": q}
    res = handoff_to_human(payload)
    return {"human_handoff": res}


def order_node(state: State) -> Dict[str, Any]:
    """订单查询节点：执行 SQL（带超时/重试，失败降级为 mock），不使用 LLM，使用确定性模板生成客服话术。"""
    q = state.get("query", "")
    payload = getdb(q)
    sql_text = None
    params = None
    try:
        spec = sql_llm.invoke(ORDER_SQL_PROMPT_TEMPLATE.format(question=q))
        sql_text = getattr(spec, "sql", None)
        params = getattr(spec, "params", None)
    except Exception:
        sql_text = None
        params = None
    def _ensure_list(x):
        return list(x) if isinstance(x, (list, tuple)) else None
    params = _ensure_list(params)
    if params:
        params = [str(v).lstrip("#") for v in params]
    def _ok_sql(s: Optional[str]) -> bool:
        t = (s or "").lower()
        return ("select" in t) and ("from orders" in t) and ("start_time" in t) and ("%s" in t)
    if not _ok_sql(sql_text) or not params:
        sql_text = payload.get("sql")
        params = payload.get("params")
    result = None
    if sql_text and params:
        try:
            runner = RunnableLambda(lambda _: exec_sql(sql_text, list(params))).with_retry()
            result = runner.invoke(None)
        except Exception:
            result = None
    if result is None:
        try:
            fb_sql = payload.get("sql")
            fb_params = payload.get("params")
            runner = RunnableLambda(lambda _: exec_sql(fb_sql, fb_params)).with_retry()
            result = runner.invoke(None)
        except Exception:
            result = None
    if result is None:
        return {"order_summary": "未查到，请输入“人工客服”进行查询"}
    need_time = any(k in q for k in ["开课", "开课时间", "什么时候开课"])
    if need_time and not (result.get("start_time") or ""):
        return {"order_summary": "未查到，请输入“人工客服”进行查询"}
    try:
        msg = llm.invoke(ORDER_NLG_PROMPT_TEMPLATE.format(
            order_id=str(result.get("order_id") or ""),
            status=str(result.get("status") or "未知"),
            amount=str(result.get("amount") if result.get("amount") is not None else "未知"),
            updated_at=str(result.get("updated_at") or ""),
            start_time=str(result.get("start_time") or ""),
        ))
        s = str(getattr(msg, "content", msg)).strip()
    except Exception:
        s = _format_order_nlg(result)
    if not s:
        s = _format_order_nlg(result)
    return {"order_summary": s}


def direct_node(state: State) -> Dict[str, Any]:
    """直答节点：不依赖 KB 的简要回答。"""
    q = state.get("query", "")
    msg = llm.invoke(DIRECT_PROMPT_TEMPLATE.format(question=q))
    content = str(getattr(msg, "content", msg)).strip()
    return {"kb_answer": content}


def decide_after_kb(state: State) -> str:
    """KB 回答后续路由：有答案直接结束，否则转人工。"""
    ans = (state.get("kb_answer") or "").strip()
    return "has_kb" if ans else "no_kb"


def build_graph():
    """构建并编译对话图，含检查点存储。"""
    g = StateGraph(State)
    g.add_node("intent", intent_node)
    g.add_node("kb", kb_node)
    g.add_node("handoff", no_kb_then_handoff_node)
    g.add_node("order", order_node)
    g.add_node("direct", direct_node)
    g.add_edge(START, "intent")
    def _branch(state: State) -> str:
        intent = state.get("intent", "direct")
        if intent in {"course", "presale", "postsale"}:
            return "kb"
        if intent == "order":
            return "order"
        if intent == "human":
            return "handoff"
        return "direct"
    g.add_conditional_edges("intent", _branch, {"kb": "kb", "order": "order", "handoff": "handoff", "direct": "direct"})
    g.add_conditional_edges("kb", decide_after_kb, {"has_kb": END, "no_kb": "handoff"})
    g.add_edge("order", END)
    g.add_edge("handoff", END)
    g.add_edge("direct", END)
    checkpointer = config.get_checkpointer()
    chain = g.compile(checkpointer=checkpointer)
    return chain
