try:
    from work.graph import build_graph
    from work.statee import State
except Exception:
    from graph import build_graph
    from statee import State

def test_keyword_human_route():
    chain = build_graph()
    out = chain.invoke({"query": "请转人工客服"}, {"configurable": {"thread_id": "t-graph"}})
    assert out.get("route") == "human"

def test_keyword_order_route():
    chain = build_graph()
    out = chain.invoke({"query": "订单 20251114001 进度"}, {"configurable": {"thread_id": "t-graph"}})
    assert out.get("route") in {"order","direct"}