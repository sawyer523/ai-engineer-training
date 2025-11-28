"""工具函数集合

包含：
- 知识库检索（FAISS）
- 订单数据库查询与兜底 mock
- 未命中问题记录（SQLite）
- 订单信息的自然语言格式化
- 转人工渠道封装
"""
import os
import time
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import config
except Exception:
    import config as config


def retrieve_kb(query: str) -> Tuple[str, List[Any]]:
    """根据查询在向量库中检索相似文档。

    返回值：
    - serialized：将检索到的文档以 "Source/Content" 形式串联的文本
    - docs：原始文档对象列表（用于提取来源 metadata）
    """
    vs = config.get_vector_store()
    docs: List[Any] = []
    if vs is not None:
        docs = vs.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {getattr(doc, 'metadata', {})}\nContent: {getattr(doc, 'page_content', '')}")
        for doc in docs
    )
    return serialized, docs


def _parse_order_id(text: str) -> Optional[str]:
    """从文本中提取订单号，支持带或不带 `#` 前缀。

    规则：匹配 3~20 位数字，统一返回形如 `#20251114001` 的格式。
    """
    import re
    m = re.search(r"#?\d{3,20}", (text or ""))
    if not m:
        return None
    s = m.group(0)
    return s if s.startswith("#") else f"#{s}"


def getdb(order_text: str) -> Dict[str, Any]:
    """生成订单查询所需的 SQL 与参数，并包含兜底的 mock 数据。"""
    oid = _parse_order_id(order_text) or "#20251114001"
    mock = {
        "order_id": oid,
        "status": "processing",
        "amount": 199.0,
        "timeline": [
            "2025-11-10 申请",
            "2025-11-12 审核中",
            "2025-11-15 等待支付确认",
        ],
    }
    sql = (
        # 注意：SQLite 使用问号占位符，这里用 %s 后续会替换为 ?
        "SELECT order_id, status, amount, updated_at, start_time FROM orders WHERE order_id = %s LIMIT 1"
    )
    params = [oid.lstrip("#")]
    return {"mock": mock, "sql": sql, "params": params}


def exec_sql(sql: str, params: List[Any]) -> Optional[Dict[str, Any]]:
    """执行订单查询 SQL 并返回结构化结果。

    当数据库路径缺失或执行失败时返回 None。
    """
    db_path = config.get_orders_db_path()
    if not db_path:
        return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # 将通用占位符 %s 替换为 SQLite 的 ?
        cur.execute(sql.replace("%s", "?"), params)
        row = cur.fetchone()
        print("exec_sql:", sql, params, row)
        cur.close()
        conn.close()
        if row:
            return {
                "order_id": str(row[0]),
                "status": str(row[1]),
                "amount": float(row[2]) if row[2] is not None else None,
                "updated_at": str(row[3]) if row[3] is not None else None,
                "start_time": str(row[4]) if len(row) > 4 and row[4] is not None else None,
            }
        return None
    except Exception:
        return None


def record_unanswered(text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """将未命中的用户问题记录到本地 SQLite，便于人工回溯。"""
    db_path = config.SUPPORT_DB_PATH or os.path.normpath(os.path.join(os.path.dirname(__file__), "support.db"))
    ts = int(time.time())
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS unanswered_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, text TEXT, created_at INTEGER)"
    )
    cur.execute(
        "INSERT INTO unanswered_questions(user_id, text, created_at) VALUES(?, ?, ?)",
        [user_id, text, ts],
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True, "db": db_path}


def _format_order_nlg(item: Dict[str, Any]) -> str:
    """将订单信息整理为中文客服话术，便于直接回复用户。"""
    oid = item.get("order_id") or ""
    status = item.get("status") or "未知"
    amount = item.get("amount")
    updated_at = item.get("updated_at")
    start_time = item.get("start_time")
    parts: List[str] = []
    first = f"您的订单 {oid} 当前状态为{status}"
    if amount is not None:
        first += f"，订单金额为{amount}元"
    parts.append(first)
    if updated_at:
        parts.append(f"最近更新时间为{updated_at}")
    if start_time:
        parts.append(f"您已成功报名参加课程，开课时间为{start_time}，请在开课前做好相关预习准备，祝您学习顺利")
    timeline = item.get("timeline")
    if isinstance(timeline, list) and timeline:
        parts.append("处理时间线：" + "，".join(timeline))
    return "。".join(parts)


def handoff_to_human(payload: Dict[str, Any]) -> Dict[str, Any]:
    """封装转人工的渠道与载荷。"""
    url = config.HUMAN_SUPPORT_URL
    return {"channel": url or "default", "payload": payload}
