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

_VECTOR_STORE: Optional[FAISS] = None


def _default_kb_index_dir() -> str:
    """返回默认的 KB 索引目录（优先使用工作目录下的 `faiss_index`）。"""
    base = os.path.dirname(__file__)
    p_work = os.path.normpath(os.path.join(base, "faiss_index"))
    if os.path.isdir(p_work):
        return p_work
    return os.path.normpath(os.path.join(base, "..", "tests", "faiss_index"))


def get_llm() -> ChatTongyi:
    """构建通义千问 Chat 模型实例。"""
    return ChatTongyi(model=MODEL_NAME)


def get_embeddings() -> DashScopeEmbeddings:
    """创建嵌入模型，用于向量检索。"""
    key = os.getenv("DASHSCOPE_API_KEY")
    return DashScopeEmbeddings(model=EMBEDDING_MODEL, dashscope_api_key=key)


def get_vector_store() -> Optional[FAISS]:
    """懒加载 FAISS 向量索引。

    如果索引目录包含非 ASCII 字符，尝试复制到临时目录以规避路径编码问题。
    """
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE
    try:
        embeddings = get_embeddings()
        index_dir = KB_INDEX_DIR or _default_kb_index_dir()
        def _ascii_dir(p: str) -> str:
            try:
                p.encode("ascii")
                return p
            except Exception:
                import tempfile
                import shutil
                base = os.path.join(tempfile.gettempdir(), "kb_index")
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
        _VECTOR_STORE = FAISS.load_local(safe_dir, embeddings, allow_dangerous_deserialization=True)
        logging.info("FAISS index loaded: %s", index_dir)
        return _VECTOR_STORE
    except Exception as e:
        logging.warning("FAISS index load failed: %s", e)
        return None


def get_orders_db_path() -> Optional[str]:
    """返回订单数据库路径，优先环境变量，其次工作目录默认路径。"""
    p = ORDERS_DB_PATH
    if p:
        return p
    base = os.path.dirname(__file__)
    default = os.path.normpath(os.path.join(base, "db", "orders.sqlite"))
    if os.path.isfile(default):
        return default
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


def get_checkpointer():
    """返回 LangGraph 的检查点存储实现。

    优先使用 SQLite（当路径可用），否则回退为内存存储。
    """
    try:
        if CHECKPOINT_DB_PATH:
            from langgraph.checkpoint.sqlite import SqliteSaver
            return SqliteSaver(CHECKPOINT_DB_PATH)
    except Exception as e:
        logging.warning("SqliteSaver unavailable, fallback to memory: %s", e)
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

def get_metrics() -> Metrics:
    global _METRICS
    if _METRICS is None:
        _METRICS = Metrics()
    return _METRICS
