import time
from fastapi.testclient import TestClient

try:
    from work.app import app
except Exception:
    from app import app

client = TestClient(app)

def test_sse_first_event():
    tid = "sse-1"
    client.post("/chat", json={"query": "课程咨询", "thread_id": tid})
    with client.stream("GET", f"/suggest/{tid}") as r:
        it = r.iter_lines()
        t0 = time.time()
        line = next(it)
        assert isinstance(line, (bytes, str))
        assert (time.time() - t0) < 1.0