import json
from fastapi.testclient import TestClient

try:
    from work.app import app
except Exception:
    from app import app

client = TestClient(app)

def test_greet_options():
    r = client.get("/greet")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data and "options" in data
    opts = {o["key"] for o in data["options"]}
    assert {"course","order","human"}.issubset(opts)