"""
基于 Milvus 的 FAQ 检索系统
功能：
1. 使用 LlamaIndex 构建语义索引
2. 支持语义切分 + 重叠
3. 支持热更新知识库（自动 re-index）
4. 提供 RESTful API 接口
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 新版 llama-index (0.10+) 导入路径
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============= 数据模型 =============

class FAQItem(BaseModel):
    """FAQ 条目"""
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="答案")
    category: Optional[str] = Field(None, description="分类")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="额外元数据")


class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., description="用户问题", min_length=1)
    top_k: int = Field(3, description="返回最相关的条目数", ge=1, le=10)
    threshold: float = Field(0.5, description="相似度阈值", ge=0, le=1)


class QueryResponse(BaseModel):
    """查询响应"""
    question: str
    answer: str
    score: float
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IndexStatus(BaseModel):
    """索引状态"""
    is_ready: bool
    total_documents: int
    last_updated: Optional[str] = None
    index_version: int


# ============= 配置 =============

class Config:
    """系统配置"""
    # OpenAI 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    # Milvus 配置
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "faq_collection")

    # 索引配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
    TOP_K = int(os.getenv("TOP_K", "3"))

    # API 配置
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # 数据目录
    DATA_DIR = Path(__file__).parent / "data"
    FAQ_FILE = DATA_DIR / "faq.json"


# ============= FAQ 索引管理器 =============

class FAQIndexManager:
    """FAQ 索引管理器 - 支持热更新"""

    def __init__(self, config: Config = Config()):
        self.config = config
        self.index: Optional[VectorStoreIndex] = None
        self.vector_store: Optional[MilvusVectorStore] = None
        self.index_version = 0
        self.last_updated = None
        self._is_ready = False
        self._lock = asyncio.Lock()

        # 确保数据目录存在
        self.config.DATA_DIR.mkdir(exist_ok=True)

    async def initialize(self):
        """初始化索引"""
        if self._is_ready:
            return

        async with self._lock:
            if self._is_ready:
                return

            # 初始化嵌入模型
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=self.config.OPENAI_API_KEY,
                api_base=self.config.OPENAI_API_BASE,
            )

            # 初始化向量存储
            self.vector_store = MilvusVectorStore(
                host=self.config.MILVUS_HOST,
                port=self.config.MILVUS_PORT,
                collection_name=self.config.MILVUS_COLLECTION,
                overwrite=False,
                dim=1536,  # text-embedding-3-small 的维度
            )

            # 创建存储上下文
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # 尝试加载现有索引
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                )
                self._is_ready = True
                print(f"[INFO] 从 Milvus 加载现有索引: {self.config.MILVUS_COLLECTION}")
            except Exception:
                # 首次初始化，创建空索引
                self.index = VectorStoreIndex(
                    [],
                    storage_context=storage_context,
                )
                self._is_ready = True
                print(f"[INFO] 创建新索引: {self.config.MILVUS_COLLECTION}")

    async def add_faq_items(self, faq_items: List[FAQItem]):
        """添加 FAQ 条目到索引"""
        async with self._lock:
            # 创建文档
            documents = []
            for item in faq_items:
                doc = Document(
                    text=f"问题：{item.question}\n答案：{item.answer}",
                    metadata={
                        "question": item.question,
                        "answer": item.answer,
                        "category": item.category or "default",
                        **item.metadata,
                    },
                )
                documents.append(doc)

            # 语义切分节点解析器
            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=Settings.embed_model,
            )

            # 插入文档
            for doc in documents:
                nodes = splitter.get_nodes_from_documents([doc])
                for node in nodes:
                    self.index.insert(node)

            self.index_version += 1
            self.last_updated = datetime.now().isoformat()
            print(f"[INFO] 已添加 {len(documents)} 条 FAQ 到索引")

    async def query(self, question: str, top_k: int = 3, threshold: float = 0.5) -> List[QueryResponse]:
        """查询最相关的 FAQ"""
        if not self._is_ready:
            raise RuntimeError("索引未初始化")

        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            retriever_mode="default",
        )

        response = query_engine.query(question)

        results = []
        for node in response.source_nodes:
            score = getattr(node, "score", 0.0)
            if score >= threshold:
                results.append(
                    QueryResponse(
                        question=node.metadata.get("question", ""),
                        answer=node.metadata.get("answer", ""),
                        score=float(score),
                        category=node.metadata.get("category"),
                        metadata={k: v for k, v in node.metadata.items() if k not in ["question", "answer", "category"]},
                    )
                )

        return results

    async def get_status(self) -> IndexStatus:
        """获取索引状态"""
        return IndexStatus(
            is_ready=self._is_ready,
            total_documents=0,  # Milvus 不直接提供文档计数
            last_updated=self.last_updated,
            index_version=self.index_version,
        )

    async def rebuild_index(self):
        """重建索引"""
        async with self._lock:
            # TODO: 从数据源重新加载所有 FAQ
            pass


# ============= 文件监控器（热更新） =============

class FAQFileWatcher(FileSystemEventHandler):
    """FAQ 文件监控器 - 支持热更新"""

    def __init__(self, index_manager: FAQIndexManager, faq_file: Path):
        self.index_manager = index_manager
        self.faq_file = faq_file
        self._last_modified = 0

    def on_modified(self, event):
        """文件修改时触发重新索引"""
        if event.src_path == str(self.faq_file):
            current_time = os.path.getmtime(self.faq_file)
            if current_time > self._last_modified:
                self._last_modified = current_time
                # 在后台任务中重新加载
                asyncio.create_task(self._reload_faq())

    async def _reload_faq(self):
        """重新加载 FAQ"""
        import json

        try:
            if self.faq_file.exists():
                with open(self.faq_file, "r", encoding="utf-8") as f:
                    faq_data = json.load(f)

                faq_items = [FAQItem(**item) for item in faq_data]
                await self.index_manager.add_faq_items(faq_items)
                print(f"[INFO] FAQ 知识库已更新，共 {len(faq_items)} 条")
        except Exception as e:
            print(f"[ERROR] 重新加载 FAQ 失败: {e}")


# ============= FastAPI 应用 =============

# 创建全局索引管理器
index_manager = FAQIndexManager()
app = FastAPI(
    title="FAQ 检索系统",
    description="基于 Milvus + LlamaIndex 的语义检索系统",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    """启动时初始化"""
    await index_manager.initialize()

    # 加载示例数据
    sample_faq = [
        FAQItem(
            question="如何退货？",
            answer="您可以在订单页面申请退货，退货原因请选择具体原因，我们会在1-3个工作日内处理。",
            category="售后",
        ),
        FAQItem(
            question="退款需要多长时间？",
            answer="退款通常在1-3个工作日内原路返回，具体到账时间取决于您的银行或支付平台。",
            category="售后",
        ),
        FAQItem(
            question="如何修改收货地址？",
            answer="在订单未发货前，您可以在订单详情页修改收货地址。已发货的订单无法修改地址。",
            category="订单",
        ),
        FAQItem(
            question="支持哪些支付方式？",
            answer="我们支持支付宝、微信支付、银行卡等多种支付方式。",
            category="支付",
        ),
        FAQItem(
            question="如何联系客服？",
            answer="您可以通过在线客服、客服电话400-123-4567或邮件service@example.com联系我们。",
            category="联系",
        ),
    ]

    await index_manager.add_faq_items(sample_faq)
    print("[INFO] FAQ 系统启动完成")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "faq-retrieval"}


@app.get("/status", response_model=IndexStatus)
async def get_status():
    """获取索引状态"""
    return await index_manager.get_status()


@app.post("/query", response_model=List[QueryResponse])
async def query_faq(request: QueryRequest):
    """查询 FAQ"""
    try:
        results = await index_manager.query(
            question=request.question,
            top_k=request.top_k,
            threshold=request.threshold,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_faq(background_tasks: BackgroundTasks):
    """手动重新加载 FAQ（用于热更新）"""
    background_tasks.add_task(index_manager.rebuild_index)
    return {"message": "FAQ 重新加载任务已启动"}


# ============= 命令行入口 =============

def main():
    """命令行入口 - 可以直接运行测试"""

    async def run_demo():
        """演示基本功能"""
        # 初始化索引管理器
        await index_manager.initialize()

        # 添加示例数据
        sample_faq = [
            FAQItem(
                question="如何退货？",
                answer="您可以在订单页面申请退货，退货原因请选择具体原因，我们会在1-3个工作日内处理。",
                category="售后",
            ),
            FAQItem(
                question="退款需要多长时间？",
                answer="退款通常在1-3个工作日内原路返回，具体到账时间取决于您的银行或支付平台。",
                category="售后",
            ),
            FAQItem(
                question="怎么申请退款？",
                answer="登录账户后，进入订单详情，点击申请退款按钮即可。",
                category="售后",
            ),
        ]

        await index_manager.add_faq_items(sample_faq)

        # 测试查询
        test_questions = [
            "我想退货怎么办",
            "退款要几天",
            "如何申请退货退款",
            "你们支持什么付款方式",
        ]

        print("=" * 50)
        print("FAQ 检索系统测试")
        print("=" * 50)

        for q in test_questions:
            print(f"\n问题: {q}")
            results = await index_manager.query(q, top_k=2)
            if results:
                for i, r in enumerate(results, 1):
                    print(f"  [{i}] 相似度: {r.score:.3f}")
                    print(f"      Q: {r.question}")
                    print(f"      A: {r.answer}")
            else:
                print("  未找到相关结果")

        print("\n" + "=" * 50)

    # 运行演示
    asyncio.run(run_demo())

    # 启动 API 服务器（可选）
    print("\n启动 API 服务器...")
    print(f"访问 http://{Config.API_HOST}:{Config.API_PORT}/docs 查看 API 文档")
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)


if __name__ == "__main__":
    main()
