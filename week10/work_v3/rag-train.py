"""RAG 索引构建脚本

从 `datas/data.txt` 读取文本，切分为小段并构建 FAISS 索引，
以支持在线相似检索。
"""
import os
import dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import tempfile
import shutil
import sys
try:
    from . import config as cfg
except Exception:
    import config as cfg

dotenv.load_dotenv()

embeddings = DashScopeEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)

BASE = Path(__file__).resolve().parent
tenant_id = None
args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "--tenant" and i + 1 < len(args):
        tenant_id = args[i + 1]
        i += 2
        continue
    i += 1
DATA_DIR = os.getenv("KB_DATA_DIR") or (cfg.get_kb_data_dir(tenant_id) if tenant_id else str(BASE / "datas"))
txt_files = list(Path(DATA_DIR).glob("**/*.txt"))
docs = []
for file in txt_files:
    loader = TextLoader(str(file), encoding="utf-8")
    docs.extend(loader.load())
docs = [d for d in docs if getattr(d, "page_content", "").strip()]


# 将文本分割为小段，便于向量化与召回
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=50
)
all_splits = text_splitter.split_documents(docs)
all_splits = [d for d in all_splits if getattr(d, "page_content", "").strip()]
if not all_splits:
    print("no valid text chunks found in:", os.path.abspath(DATA_DIR))
    sys.exit(1)

try:
    vector_store = FAISS.from_documents(all_splits, embeddings)
    INDEX_DIR_PATH = Path(cfg.get_kb_index_dir(tenant_id) if tenant_id else str(BASE / "faiss_index"))
    INDEX_DIR_PATH.mkdir(parents=True, exist_ok=True)
    ascii_dir = Path(tempfile.gettempdir()) / "kb_index_train"
    ascii_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(ascii_dir))
    for name in ("index.faiss", "index.pkl"):
        src = ascii_dir / name
        dst = INDEX_DIR_PATH / name
        if src.exists():
            shutil.copyfile(str(src), str(dst))
except Exception as e:
    print("faiss index build failed:", e)
    sys.exit(1)

print("faiss_index created at:", str(INDEX_DIR_PATH))
