import os
import sys
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from work import config

def run_query(q: str, k: int = 2):
    vs = config.get_vector_store()

    results = vs.similarity_search_with_score(q, k=k)
    for doc, score in results:
        print(score)
        print(getattr(doc, "page_content", ""))
        print(getattr(doc, "metadata", {}))
    print("-----------------")


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def add_text_unique(vs, text: str, metadata=None):
    tid = _stable_id(text)
    ds = getattr(vs, "docstore", None)
    exists = False
    try:
        exists = ds and getattr(ds, "_dict", None) is not None and tid in ds._dict
    except Exception:
        exists = False
    if exists:
        print("已存在，跳过新增")
        return False, tid
    try:
        m = dict(metadata) if isinstance(metadata, dict) else {"source": "tests"}
        m["id"] = tid
        vs.add_texts([text], metadatas=[m], ids=[tid])
        print("已添加新文本到向量库")
        return True, tid
    except Exception as e:
        print("新增文本失败:", e)
        return False, None

def delete_text_by_content(vs, text: str, delete_all: bool = False):
    try:
        ds = getattr(vs, "docstore", None)
        if not ds or not getattr(ds, "_dict", None):
            print("删除失败：docstore不可用")
            return False
        matches = []
        for doc_id, doc in ds._dict.items():
            content = getattr(doc, "page_content", "")
            if text in content:
                mid = (getattr(doc, "metadata", {}) or {}).get("id") or doc_id
                matches.append(mid)
        if not matches:
            print("未找到匹配文本")
            return False
        if hasattr(vs, "delete"):
            ids_to_delete = matches if delete_all else [matches[0]]
            vs.delete(ids=ids_to_delete)
            print(f"已删除指定文本（按metadata['id']），共删除 {len(ids_to_delete)} 条")
            return True
        print("删除失败：当前向量库不支持按ID删除")
        return False
    except Exception as e:
        print("删除失败:", e)
        return False

if __name__ == "__main__":
    q = "课程会讲向量数据库吗？"
    run_query(q, k=2)
    vs = config.get_vector_store()
    new_answer = (
        "101. **Q：课程会讲解哪种向量数据库？**  \n"
        "A：课程已包含 FAISS 的向量检索，支持自然语言查询并返回最相关的问答片段及来源路径。"
    )
    added, tid = add_text_unique(vs, new_answer)
    run_query(q, k=2)
    if added and tid:
        delete_text_by_content(vs, new_answer)
        run_query(q, k=2)
