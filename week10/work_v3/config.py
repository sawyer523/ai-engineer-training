"""配置与资源加载模块

负责：
- 读取环境变量与默认值
- 初始化对话模型与向量检索组件
- 提供向量索引、订单数据库与图检查点的获取函数
"""
import os
import logging
from typing import Optional
import dotenv
from collections import deque
import threading
import json
import time
from datetime import datetime

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import InMemorySaver

dotenv.load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "qwen-turbo")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

os.environ["LANGCHAIN_PROJECT"] = "edu-agent"
os.environ["LANGCHAIN_TRACING"] = "true"

KB_INDEX_DIR = os.getenv("KB_INDEX_DIR")
ORDERS_DB_PATH = os.getenv("ORDERS_DB_PATH")
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH")
SUPPORT_DB_PATH = os.getenv("SUPPORT_DB_PATH")
HUMAN_SUPPORT_URL = os.getenv("HUMAN_SUPPORT_URL")
TENANTS_BASE_DIR = os.getenv("TENANTS_BASE_DIR")
COURSE_TENANT_MAP = os.getenv("COURSE_TENANT_MAP")

_VECTOR_STORE: Optional[FAISS] = None
_VECTOR_STORES: dict = {}
SUPPORTED_MODELS = ("qwen-turbo", "qwen-plus", "qwen-vl-max")
DEFAULT_MODEL_PARAMS = {"qwen-turbo": {}, "qwen-plus": {}, "qwen-vl-max": {}}
_CURRENT_MODEL = MODEL_NAME
_MODEL_LOCK = threading.RLock()


def _default_kb_index_dir() -> str:
    """返回默认的 KB 索引目录（优先使用工作目录下的 `faiss_index`）。"""
    base = os.path.dirname(__file__)
    p_work = os.path.normpath(os.path.join(base, "faiss_index"))
    if os.path.isdir(p_work):
        return p_work
    return os.path.normpath(os.path.join(base, "..", "tests", "faiss_index"))

def _base_dir() -> str:
    return os.path.dirname(__file__)

def _norm_tenant(tid: Optional[str]) -> str:
    t = str(tid or "").strip()
    if not t:
        return "default"
    import re
    if re.fullmatch(r"[A-Za-z0-9_]{1,64}", t):
        return t
    return "default"

def _tenants_root() -> str:
    base = _base_dir()
    if TENANTS_BASE_DIR:
        p = TENANTS_BASE_DIR
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(base, p))
        return p
    return os.path.normpath(os.path.join(base, "tenants"))

def _tenant_dir(tid: Optional[str]) -> str:
    t = _norm_tenant(tid)
    return os.path.normpath(os.path.join(_tenants_root(), t))

def get_kb_index_dir(tenant_id: Optional[str] = None) -> str:
    if tenant_id is None and KB_INDEX_DIR:
        return KB_INDEX_DIR
    if tenant_id is None:
        return _default_kb_index_dir()
    p = os.path.join(_tenant_dir(tenant_id), "faiss_index")
    return p

def get_kb_data_dir(tenant_id: Optional[str] = None) -> str:
    if tenant_id is None:
        return os.path.normpath(os.path.join(_base_dir(), "datas"))
    return os.path.join(_tenant_dir(tenant_id), "datas")


def get_llm() -> ChatTongyi:
    return ChatTongyi(model=_CURRENT_MODEL)


def get_embeddings() -> DashScopeEmbeddings:
    """创建嵌入模型，用于向量检索。"""
    key = os.getenv("DASHSCOPE_API_KEY")
    return DashScopeEmbeddings(model=EMBEDDING_MODEL, dashscope_api_key=key)


def get_vector_store(tenant_id: Optional[str] = None) -> Optional[FAISS]:
    """懒加载 FAISS 向量索引。

    如果索引目录包含非 ASCII 字符，尝试复制到临时目录以规避路径编码问题。
    """
    global _VECTOR_STORE, _VECTOR_STORES
    if tenant_id is None and _VECTOR_STORE is not None:
        return _VECTOR_STORE
    tkey = _norm_tenant(tenant_id) if tenant_id is not None else None
    if tkey is not None:
        vs_cached = _VECTOR_STORES.get(tkey)
        if vs_cached is not None:
            return vs_cached
    try:
        embeddings = get_embeddings()
        index_dir = get_kb_index_dir(tenant_id)
        def _ascii_dir(p: str) -> str:
            try:
                p.encode("ascii")
                return p
            except Exception:
                import tempfile
                import shutil
                base = os.path.join(tempfile.gettempdir(), "kb_index" + (f"_{tkey}" if tkey else ""))
                os.makedirs(base, exist_ok=True)
                for name in ("index.faiss", "index.pkl"):
                    src = os.path.join(p, name)
                    dst = os.path.join(base, name)
                    try:
                        shutil.copyfile(src, dst)
                    except Exception:
                        pass
                return base
        safe_dir = _ascii_dir(index_dir)
        vs = FAISS.load_local(safe_dir, embeddings, allow_dangerous_deserialization=True)
        logging.info("FAISS index loaded: %s", index_dir)
        if tkey is None:
            _VECTOR_STORE = vs
        else:
            _VECTOR_STORES[tkey] = vs
        return vs
    except Exception as e:
        logging.warning("FAISS index load failed: %s", e)
        return None


def get_orders_db_path(tenant_id: Optional[str] = None) -> Optional[str]:
    """返回订单数据库路径，优先环境变量，其次工作目录默认路径。"""
    p = ORDERS_DB_PATH
    if p and tenant_id is None:
        return p
    if tenant_id is None:
        base = os.path.dirname(__file__)
        default = os.path.normpath(os.path.join(base, "db", "orders.sqlite"))
        if os.path.isfile(default):
            return default
    else:
        default = os.path.normpath(os.path.join(_tenant_dir(tenant_id), "db", "orders.sqlite"))
        if os.path.isfile(default):
            return default
        try:
            base_default = os.path.normpath(os.path.join(_base_dir(), "db", "orders.sqlite"))
            if os.path.isfile(base_default):
                import shutil
                os.makedirs(os.path.dirname(default), exist_ok=True)
                shutil.copyfile(base_default, default)
                return default
        except Exception:
            pass
    try:
        try:
            from . import init_orders_db as _init
        except Exception:
            import init_orders_db as _init
        _ = _init  # ensure imported
        if os.path.isfile(default):
            return default
    except Exception:
        pass
    return None


def get_checkpointer(tenant_id: Optional[str] = None):
    """返回 LangGraph 的检查点存储实现。

    优先使用 SQLite（当路径可用），否则回退为内存存储。
    """
    try:
        if CHECKPOINT_DB_PATH and tenant_id is None:
            from langgraph.checkpoint.sqlite import SqliteSaver
            return SqliteSaver(CHECKPOINT_DB_PATH)
    except Exception as e:
        logging.warning("SqliteSaver unavailable, fallback to memory: %s", e)
    if tenant_id is not None:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            p = os.path.normpath(os.path.join(_tenant_dir(tenant_id), "checkpoints", "sqlite.db"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            return SqliteSaver(p)
        except Exception:
            pass
    return InMemorySaver()

class _Stats:
    def __init__(self, maxlen: int = 1000):
        self.lock = threading.Lock()
        self.window = deque(maxlen=maxlen)
        self.count = 0
        self.sum = 0.0
        self.min = None
        self.max = None
    def update(self, v: float):
        with self.lock:
            self.window.append(v)
            self.count += 1
            self.sum += v
            self.min = v if self.min is None or v < self.min else self.min
            self.max = v if self.max is None or v > self.max else self.max
    def snapshot(self) -> dict:
        with self.lock:
            n = self.count
            avg = (self.sum / n) if n else 0.0
            mn = self.min if self.min is not None else 0.0
            mx = self.max if self.max is not None else 0.0
            arr = list(self.window)
        p95 = 0.0
        if arr:
            arr.sort()
            idx = max(int(len(arr) * 0.95) - 1, 0)
            p95 = arr[idx]
        return {"count": n, "min_ms": mn, "max_ms": mx, "avg_ms": avg, "p95_ms": p95}

class Metrics:
    def __init__(self):
        self._stats = {}
        self._lock = threading.Lock()
    def update(self, key: str, v: float):
        with self._lock:
            s = self._stats.get(key)
            if s is None:
                s = _Stats()
                self._stats[key] = s
        s.update(v)
    def snapshot(self, key: str) -> dict:
        s = self._stats.get(key)
        return s.snapshot() if s else {"count": 0, "min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}
    def snapshot_all(self) -> dict:
        with self._lock:
            keys = list(self._stats.keys())
        return {k: self.snapshot(k) for k in keys}

_METRICS: Optional[Metrics] = None
_REDIS = None
_SESSIONS = {}

def get_metrics() -> Metrics:
    global _METRICS
    if _METRICS is None:
        _METRICS = Metrics()
    return _METRICS

def get_redis():
    global _REDIS
    if _REDIS is not None:
        return _REDIS
    try:
        import redis
        url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        _REDIS = redis.Redis.from_url(url)
        _REDIS.ping()
        return _REDIS
    except Exception:
        _REDIS = None
        return None

def _sess_key(thread_id: str) -> str:
    return f"session:{thread_id}"

def get_session_messages(thread_id: str, maxlen: int = 5) -> list:
    r = get_redis()
    if r is not None:
        try:
            key = _sess_key(thread_id)
            vals = r.lrange(key, -maxlen, -1)
            out = []
            for v in vals:
                try:
                    out.append(json.loads(v))
                except Exception:
                    pass
            return out
        except Exception:
            pass
    now = int(time.time())
    sess = _SESSIONS.get(thread_id) or {"ts": now, "arr": []}
    arr = sess["arr"][-maxlen:]
    return list(arr)

def append_session_message(thread_id: str, role: str, content: str, ttl_seconds: int = 1800, maxlen: int = 5):
    item = {"role": role, "content": content}
    r = get_redis()
    if r is not None:
        try:
            key = _sess_key(thread_id)
            r.rpush(key, json.dumps(item, ensure_ascii=False))
            r.ltrim(key, -maxlen, -1)
            r.expire(key, ttl_seconds)
            return True
        except Exception:
            pass
    now = int(time.time())
    sess = _SESSIONS.get(thread_id)
    if sess is None:
        sess = {"ts": now, "arr": []}
        _SESSIONS[thread_id] = sess
    sess["ts"] = now
    arr = sess["arr"]
    arr.append(item)
    if len(arr) > maxlen:
        del arr[0:len(arr)-maxlen]
    return True

def reset_session(thread_id: str):
    r = get_redis()
    if r is not None:
        try:
            r.delete(_sess_key(thread_id))
            return True
        except Exception:
            pass
    _SESSIONS.pop(thread_id, None)
    return True

def get_supported_models() -> list:
    return list(SUPPORTED_MODELS)

_TENANT_MODELS: dict = {}
_COURSE_MAP_CACHE: Optional[dict] = None

def get_current_model_name(tenant_id: Optional[str] = None) -> str:
    if tenant_id is None:
        return _CURRENT_MODEL
    t = _norm_tenant(tenant_id)
    v = _TENANT_MODELS.get(t)
    return v or _CURRENT_MODEL

def validate_model(name: str) -> dict:
    n = str(name or "").strip()
    if n not in SUPPORTED_MODELS:
        return {"ok": False, "code": "unsupported", "message": "模型不受支持"}
    try:
        _ = ChatTongyi(model=n)
        return {"ok": True, "code": "ok", "message": "可用"}
    except Exception as e:
        return {"ok": False, "code": "init_error", "message": str(e)}

def switch_model(name: str, tenant_id: Optional[str] = None) -> dict:
    n = str(name or "").strip()
    with _MODEL_LOCK:
        v = validate_model(n)
        if not v.get("ok"):
            return {"ok": False, "code": v.get("code"), "message": v.get("message")}
        if tenant_id is None:
            old = _CURRENT_MODEL
            if n == old:
                return {"ok": True, "code": "noop", "message": "已是当前模型", "old": old, "new": old}
            globals()["MODEL_NAME"] = n
            globals()["_CURRENT_MODEL"] = n
            logging.info("model switched: %s -> %s", old, n)
            try:
                base = os.path.dirname(__file__)
                logs = os.path.join(base, "logs")
                os.makedirs(logs, exist_ok=True)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                line = f"{ts} | INFO | root | model switch old={old} new={n}"
                with open(os.path.join(logs, "requests.log"), "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
            return {"ok": True, "code": "switched", "message": "切换成功", "old": old, "new": n}
        t = _norm_tenant(tenant_id)
        old = _TENANT_MODELS.get(t) or _CURRENT_MODEL
        _TENANT_MODELS[t] = n
        logging.info("tenant model switched: %s -> %s for %s", old, n, t)
        try:
            base = os.path.dirname(__file__)
            logs = os.path.join(base, "logs")
            os.makedirs(logs, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            line = f"{ts} | INFO | root | tenant model switch tenant={t} old={old} new={n}"
            with open(os.path.join(logs, "requests.log"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        return {"ok": True, "code": "switched", "message": "切换成功", "old": old, "new": n}

def get_model_lock():
    return _MODEL_LOCK

def get_support_db_path(tenant_id: Optional[str] = None) -> str:
    if tenant_id is None and SUPPORT_DB_PATH:
        return SUPPORT_DB_PATH
    if tenant_id is None:
        return os.path.normpath(os.path.join(_base_dir(), "support.db"))
    return os.path.normpath(os.path.join(_tenant_dir(tenant_id), "support.db"))

def _course_map_path() -> str:
    if COURSE_TENANT_MAP:
        p = COURSE_TENANT_MAP
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(_base_dir(), p))
        return p
    return os.path.normpath(os.path.join(_base_dir(), "tenant_courses.json"))

def load_course_tenant_map() -> dict:
    global _COURSE_MAP_CACHE
    if _COURSE_MAP_CACHE is not None:
        return _COURSE_MAP_CACHE
    p = _course_map_path()
    out = {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        arr = data.get("courses") if isinstance(data, dict) else []
        for item in arr or []:
            name = str(item.get("name") or "").strip().lower()
            if not name:
                continue
            out[name] = {
                "tenant_id": str(item.get("tenant_id") or "default").strip(),
                "orders_db": str(item.get("orders_db") or "").strip(),
                "kb_index": str(item.get("kb_index") or "").strip(),
                "kb_data": str(item.get("kb_data") or "").strip(),
            }
    except Exception:
        out = {}
    _COURSE_MAP_CACHE = out
    return out

def get_tenant_for_course(course_name: str) -> str:
    name = str(course_name or "").strip().lower()
    m = load_course_tenant_map()
    info = m.get(name)
    t = str(info.get("tenant_id") if isinstance(info, dict) else "")
    return _norm_tenant(t) if t else "default"

def get_paths_for_course(course_name: str) -> dict:
    t = get_tenant_for_course(course_name)
    return {
        "tenant_id": t,
        "orders_db_path": get_orders_db_path(t),
        "kb_index_dir": get_kb_index_dir(t),
        "kb_data_dir": get_kb_data_dir(t),
        "support_db_path": get_support_db_path(t),
    }
