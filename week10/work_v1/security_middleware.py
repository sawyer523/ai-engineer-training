# python /d:/AI工程化训练营/workspace/week10/work/security_middleware.py
import re, os, json, logging
from typing import Any, Dict, List, Optional, Set, Tuple
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class SensitiveConfig:
    def __init__(self, fields: Set[str], patterns: List[re.Pattern], value_patterns: List[re.Pattern], max_body_bytes: int = 1048576):
        self.fields = {s.strip().lower() for s in fields if s and s.strip()}
        self.patterns = patterns
        self.value_patterns = value_patterns
        self.max_body_bytes = max_body_bytes

def build_default_config() -> SensitiveConfig:
    env_fields = {x.strip().lower() for x in (os.getenv("SENSITIVE_FIELDS", "").split(",") if os.getenv("SENSITIVE_FIELDS") else [])}
    default_fields = {"password","passwd","pwd","secret","token","id_number","身份证","card_no","银行卡号","bank_card","密码"}
    fields = default_fields.union(env_fields)
    patterns = [
        re.compile(r"\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]\b"),
        re.compile(r"(?<!\d)\d{15}(?!\d)"),
        re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)"),
    ]
    value_patterns = [
        re.compile(r"(?i)(密码|passw(?:or)?d|pass|pwd)\s*[:=：]?\s*(\S{4,})"),
        re.compile(r"(?i)(id[_\- ]?number|身份证)\s*[:=：]?\s*(\d{15,18}[0-9Xx]?)"),
        re.compile(r"(?i)(card[_\- ]?no|bank[_\- ]?card|银行卡号)\s*[:=：]?\s*((?:\d[ -]?){16,19})"),
    ]
    mbs = int(os.getenv("REDACT_MAX_BODY", "1048576"))
    return SensitiveConfig(fields, patterns, value_patterns, mbs)

def _redact_text(s: str, cfg: SensitiveConfig) -> Tuple[str, int]:
    count = 0
    out = s
    for p in cfg.patterns:
        out, n = p.subn("[REDACTED]", out)
        count += n
    for p in cfg.value_patterns:
        def repl(m):
            nonlocal count
            count += 1
            return f"{m.group(1)} [REDACTED]"
        out = p.sub(repl, out)
    return out, count

def _sanitize_obj(obj: Any, cfg: SensitiveConfig, changed: List[str]) -> Tuple[Any, int]:
    redactions = 0
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in cfg.fields:
                out[k] = "[REDACTED]"
                changed.append(lk)
                redactions += 1
            else:
                nv, n = _sanitize_obj(v, cfg, changed)
                out[k] = nv
                redactions += n
        return out, redactions
    if isinstance(obj, list):
        out = []
        for v in obj:
            nv, n = _sanitize_obj(v, cfg, changed)
            out.append(nv)
            redactions += n
        return out, redactions
    if isinstance(obj, tuple):
        arr = []
        for v in obj:
            nv, n = _sanitize_obj(v, cfg, changed)
            arr.append(nv)
            redactions += n
        return tuple(arr), redactions
    if isinstance(obj, str):
        ns, n = _redact_text(obj, cfg)
        return ns, n
    return obj, 0

def _sanitize_json_bytes(body: bytes, cfg: SensitiveConfig) -> Tuple[bytes, int, List[str]]:
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return body, 0, []
    changed: List[str] = []
    out, n = _sanitize_obj(data, cfg, changed)
    try:
        return json.dumps(out, ensure_ascii=False).encode("utf-8"), n, changed
    except Exception:
        return body, 0, []

def _is_json(ct: Optional[str]) -> bool:
    t = (ct or "").lower()
    return "application/json" in t or "json" in t

def sanitize_text(s: str, cfg: Optional[SensitiveConfig] = None) -> str:
    c = cfg or build_default_config()
    return _redact_text(s, c)[0]

def sanitize_dict(d: Dict[str, Any], cfg: Optional[SensitiveConfig] = None) -> Dict[str, Any]:
    c = cfg or build_default_config()
    out, _ = _sanitize_obj(d, c, [])
    return out

class RedactionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: Optional[SensitiveConfig] = None):
        super().__init__(app)
        self.cfg = config or build_default_config()

    async def dispatch(self, request: Request, call_next):
        try:
            req_body = await request.body()
        except Exception:
            req_body = b""
        req_ct = request.headers.get("content-type")
        if _is_json(req_ct) and req_body and len(req_body) <= self.cfg.max_body_bytes:
            s_body, s_cnt, s_keys = _sanitize_json_bytes(req_body, self.cfg)
            if s_cnt:
                logging.info("redact req count=%d keys=%s", s_cnt, ",".join(sorted(set(s_keys))))
            req_body = s_body
        async def receive():
            return {"type": "http.request", "body": req_body, "more_body": False}
        request = Request(request.scope, receive)
        response = await call_next(request)
        try:
            resp_ct = response.headers.get("content-type")
            raw = getattr(response, "body", b"") or b""
            if _is_json(resp_ct) and raw and len(raw) <= self.cfg.max_body_bytes:
                s_raw, s_cnt, s_keys = _sanitize_json_bytes(raw, self.cfg)
                if s_cnt:
                    logging.info("redact resp count=%d keys=%s", s_cnt, ",".join(sorted(set(s_keys))))
                    headers = dict(response.headers)
                    headers.pop("content-length", None)
                    return Response(content=s_raw, status_code=response.status_code, headers=headers, media_type=response.media_type, background=response.background)
        except Exception:
            pass
        return response