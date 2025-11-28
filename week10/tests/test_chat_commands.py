from fastapi.testclient import TestClient

try:
    from work.app import app
    from work import config
except Exception:
    from app import app
    import config

client = TestClient(app)

def test_help_command():
    r = client.post("/chat", json={"query": "/help"})
    assert r.status_code == 200
    data = r.json()
    assert any(c.get("cmd") == "/help" for c in data.get("commands", []))

def test_reset_and_history():
    tid = "t1"
    client.post("/chat", json={"query": "你好", "thread_id": tid})
    r = client.post("/chat", json={"query": "/history", "thread_id": tid})
    assert r.status_code == 200
    hist = r.json().get("history", [])
    assert len(hist) >= 1
    r2 = client.post("/chat", json={"query": "/reset", "thread_id": tid})
    assert r2.status_code == 200
    r3 = client.post("/chat", json={"query": "/history", "thread_id": tid})
    assert r3.status_code == 200
    hist2 = r3.json().get("history", [])
    assert hist2 == []