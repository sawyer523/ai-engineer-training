import time
from fastapi.testclient import TestClient

try:
    from work.app import app
    from work import config
    from work import graph as graph
except Exception:
    from app import app
    import config
    import graph as graph

client = TestClient(app)

def test_models_list_contains_supported():
    t0 = time.perf_counter()
    r = client.get("/models/list")
    dt = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 200
    data = r.json()
    assert data.get("code") == 0
    items = data.get("data", {}).get("models", [])
    assert set(["qwen-turbo", "qwen-plus"]).issubset(set(items))
    assert dt < 100.0

def test_models_switch_invalid():
    r = client.post("/models/switch", json={"name": "not-a-model"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("code") != 0

def test_models_switch_success_and_effective():
    r1 = client.post("/models/switch", json={"name": "qwen-plus"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1.get("code") == 0
    cur = d1.get("data", {}).get("current")
    assert cur == "qwen-plus"
    assert getattr(graph.llm, "model_name", None) == "qwen-plus"
    r2 = client.post("/models/switch", json={"name": "qwen-turbo"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2.get("code") == 0
    cur2 = d2.get("data", {}).get("current")
    assert cur2 == "qwen-turbo"