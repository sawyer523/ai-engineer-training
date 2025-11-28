import os
import dotenv
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None
try:
    from langchain_community.document_loaders import Docx2txtLoader
except Exception:
    Docx2txtLoader = None
try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
except Exception:
    UnstructuredExcelLoader = None

dotenv.load_dotenv()

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# Load and chunk contents of the blog
base = Path(__file__).resolve().parent
data_dir = os.getenv("KB_DATA_DIR") or str(base)
docs = []
for p in Path(data_dir).glob("**/*"):
    sfx = p.suffix.lower()
    if sfx == ".txt":
        docs.extend(TextLoader(str(p), encoding="utf-8").load())
    elif sfx == ".pdf" and PyPDFLoader:
        docs.extend(PyPDFLoader(str(p)).load())
    elif sfx == ".docx" and Docx2txtLoader:
        docs.extend(Docx2txtLoader(str(p)).load())
    elif sfx in (".xlsx", ".xls") and UnstructuredExcelLoader:
        docs.extend(UnstructuredExcelLoader(str(p)).load())
if not docs and Path("./data.txt").exists():
    docs = TextLoader("./data.txt", encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(all_splits, embeddings)

# 持久化向量数据库
vector_store.save_local("faiss_index")
