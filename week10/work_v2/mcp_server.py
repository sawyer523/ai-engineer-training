from typing import Any, Dict, List, Optional
import os

try:
    from .tools import retrieve_kb, getdb, exec_sql, _format_order_nlg, load_course_catalog
    from .graph import build_graph
except Exception:
    from tools import retrieve_kb, getdb, exec_sql, _format_order_nlg, load_course_catalog
    from graph import build_graph

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EduAgent MCP")

@mcp.tool()
def kb_search(query: str, k: int = 2) -> Dict[str, Any]:
    serialized, docs = retrieve_kb(query)
    try:
        sources = [getattr(d, "metadata", {}) for d in (docs or [])]
    except Exception:
        sources = []
    return {"context": serialized, "sources": sources[:k]}

@mcp.tool()
def order_lookup(text: str) -> Dict[str, Any]:
    payload = getdb(text)
    sql = payload.get("sql")
    params = payload.get("params")
    result = None
    try:
        result = exec_sql(sql, params)
    except Exception:
        result = None
    if result is None:
        result = payload.get("mock") or {}
    return result

@mcp.tool()
def course_catalog(limit: int = 20) -> Dict[str, Any]:
    data = load_course_catalog()
    items = list(data.get("items", []))[:limit]
    return {"sections": list(data.get("sections", [])), "items": items}

@mcp.tool()
def chat(query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    chain = build_graph()
    state = {"query": (query or "").strip(), "history": ""}
    cfg = {"configurable": {"thread_id": thread_id or "mcp"}}
    result = chain.invoke(state, cfg)
    route = result.get("route") or result.get("intent")
    sources = result.get("sources")
    order = result.get("order_summary")
    kb = result.get("kb_answer")
    human = result.get("human_handoff")
    answer = order or human or kb or ""
    return {"route": route, "answer": answer, "sources": sources}

@mcp.resource("kb://{query}")
def kb_resource(query: str) -> str:
    serialized, _ = retrieve_kb(query)
    return serialized

@mcp.resource("orders://{order_id}")
def order_resource(order_id: str) -> Dict[str, Any]:
    p = getdb(order_id)
    res = exec_sql(p.get("sql"), p.get("params"))
    if res is None:
        res = p.get("mock") or {}
    return res

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    port = int(os.getenv("MCP_PORT", "6278"))
    if transport == "http":
        mcp.run(transport="http", host="0.0.0.0", port=port)
    elif transport == "sse":
        mcp.run(transport="sse", host="0.0.0.0", port=port)
    else:
        mcp.run()