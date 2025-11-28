from fastapi.testclient import TestClient

try:
    from work.app import app
    from work.security_middleware import sanitize_text, sanitize_dict
except Exception:
    from app import app
    from security_middleware import sanitize_text, sanitize_dict

def test_redact_text():
    s = "身份证 510123199912120011 密码: Abcd1234"
    out = sanitize_text(s)
    assert "[REDACTED]" in out

def test_redact_dict():
    d = {"password": "secret", "note": "银行卡号 6216 1234 5678 9012"}
    out = sanitize_dict(d)
    assert out["password"] == "[REDACTED]"
    assert "[REDACTED]" in out["note"]