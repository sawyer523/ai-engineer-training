"""
融合文档检索、图谱推理的多跳问答系统 (GraphRAG)

功能：
1. 使用 Neo4j 构建企业股权图谱
2. 使用 LlamaIndex 实现文档检索
3. 实现多跳查询逻辑（Cypher + LLM 协同）
4. 构建可解释性输出（展示推理路径）
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import List, Dict, Any

# 新版 llama-index (0.10+) 导入路径
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.embeddings.openai import OpenAIEmbedding

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()

# ============= 配置 =============
class Config:
    """系统配置"""
    # OpenAI 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Neo4j 配置
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    # API 配置
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # 数据目录
    DATA_DIR = Path(__file__).parent / "data"
    INDEX_DIR = Path(__file__).parent / "vector_index"

    COMPANY_DOC_PATH = DATA_DIR / "companies.txt"
    SHAREHOLDER_CSV_PATH = DATA_DIR / "shareholders.csv"


# ============= 全局变量 =============
_rag_query_engine = None
_kg_query_engine = None
_graph_store = None
_config = Config()


# ============= 初始化 LlamaIndex Settings =============
def _initialize_settings():
    """初始化 LlamaIndex 全局设置"""
    if not _config.OPENAI_API_KEY:
        print("[WARN] OPENAI_API_KEY 未设置，将使用默认配置")

    Settings.llm = OpenAILLM(
        model=_config.OPENAI_MODEL,
        api_key=_config.OPENAI_API_KEY,
        api_base=_config.OPENAI_API_BASE,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=_config.OPENAI_API_KEY,
        api_base=_config.OPENAI_API_BASE,
    )
    print("[INFO] LlamaIndex Settings 初始化完成")


# ============= RAG 索引初始化 =============
def _initialize_rag_index():
    """初始化或加载 RAG 的向量索引"""
    global _rag_query_engine

    _config.DATA_DIR.mkdir(exist_ok=True)
    _config.INDEX_DIR.mkdir(exist_ok=True)

    # 如果没有文档，创建示例文档
    if not _config.COMPANY_DOC_PATH.exists():
        _create_sample_documents()

    if not _config.INDEX_DIR.exists() or not list(_config.INDEX_DIR.iterdir()):
        print("[INFO] 未找到向量索引，正在从文件创建...")
        documents = SimpleDirectoryReader(
            input_files=[str(_config.COMPANY_DOC_PATH)]
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=str(_config.INDEX_DIR))
        print(f"[INFO] 向量索引已创建并保存在 '{_config.INDEX_DIR}'")
    else:
        print(f"[INFO] 从 '{_config.INDEX_DIR}' 加载现有向量索引...")
        storage_context = StorageContext.from_defaults(persist_dir=str(_config.INDEX_DIR))
        index = load_index_from_storage(storage_context)
        print("[INFO] 向量索引加载成功")

    _rag_query_engine = index.as_query_engine(similarity_top_k=2)


def _create_sample_documents():
    """创建示例文档"""
    sample_text = """公司名称: 星辰科技
成立日期: 2015-03-10
总部地点: 中国，深圳
主营业务: 人工智能解决方案、云计算服务、大数据分析平台。
公司简介: 星辰科技是一家专注于前沿AI技术研发与应用的高科技企业。自成立以来，公司致力于为全球客户提供领先的智能化产品和服务，其核心产品"天枢AI平台"在业界享有盛誉。

公司名称: 启明资本
成立日期: 2008-06-18
总部地点: 中国，北京
主营业务: 风险投资、股权投资。
公司简介: 启明资本是一家顶级的投资机构，专注于投资高科技、高成长性的创新企业。其投资组合遍布人工智能、生物医药和新能源等多个领域，成功孵化了多家上市公司。

公司名称: 红杉基金
成立日期: 2010-05-20
总部地点: 中国，上海
主营业务: 私募股权投资。
公司简介: 红杉基金是一家知名的私募股权投资机构，专注于TMT、医疗健康和消费服务等领域的投资。

公司名称: 李明
身份: 星辰科技创始人兼CEO
简介: 李明是星辰科技的创始人，拥有超过15年的人工智能领域研发经验，曾在多家知名科技公司担任技术总监。

公司名称: 王伟
身份: 星辰科技CTO
简介: 王伟负责星辰科技的技术研发工作，是深度学习领域的专家。
"""
    with open(_config.COMPANY_DOC_PATH, "w", encoding="utf-8") as f:
        f.write(sample_text)
    print(f"[INFO] 已创建示例文档: {_config.COMPANY_DOC_PATH}")


# ============= 知识图谱查询引擎初始化 =============
def _initialize_kg_query_engine():
    """初始化知识图谱查询引擎"""
    global _kg_query_engine, _graph_store
    print("[INFO] 正在连接到 Neo4j 并初始化知识图谱查询引擎...")

    _graph_store = Neo4jGraphStore(
        username=_config.NEO4J_USER,
        password=_config.NEO4J_PASSWORD,
        url=_config.NEO4J_URI,
        database=_config.NEO4J_DATABASE,
    )

    _kg_query_engine = KnowledgeGraphQueryEngine(
        storage_context=StorageContext.from_defaults(graph_store=_graph_store),
        llm=Settings.llm,
        verbose=True,
    )
    print("[INFO] 知识图谱查询引擎初始化完成")


# ============= 多跳查询主函数 =============
def multi_hop_query(question: str) -> Dict[str, Any]:
    """
    执行多跳查询：RAG -> KG -> LLM
    """
    if not _rag_query_engine or not _kg_query_engine:
        raise RuntimeError("查询引擎未初始化。请先运行初始化函数。")

    reasoning_path = []

    # 1. RAG 检索：识别问题中的核心实体
    entity_extraction_prompt = PromptTemplate(
        "从以下问题中提取出公司或机构的名称：'{question}'\n"
        "只返回名称，不要添加任何其他文字。"
    )
    formatted_prompt = entity_extraction_prompt.format(question=question)
    entity_name_response = Settings.llm.complete(formatted_prompt)
    entity_name = entity_name_response.text.strip()

    reasoning_path.append(f"步骤 1: 从问题 '{question}' 中识别出核心实体 -> '{entity_name}'")

    # 2. 图谱查询
    cypher_query = ""
    if "最大股东" in question or "控股" in question:
        cypher_query = f"""
        MATCH (shareholder:Entity)-[r:HOLDS_SHARES_IN]->(company:Entity {{name: '{entity_name}'}})
        RETURN shareholder.name AS shareholder, r.share_percentage AS percentage
        ORDER BY percentage DESC
        LIMIT 1
        """
        reasoning_path.append(f"步骤 2: 识别到关键词'最大股东'，构造精确 Cypher 查询")

        # 直接执行 Cypher
        graph_response = _graph_store.query(cypher_query)
        kg_result_text = str(graph_response)
    else:
        reasoning_path.append(f"步骤 2: 使用 LLM 将自然语言转换为 Cypher 查询")
        kg_response = _kg_query_engine.query(f"查询与 '{entity_name}' 相关的信息")
        kg_result_text = kg_response.response

    reasoning_path.append(f"   - 图谱查询结果: {kg_result_text}")

    # 3. RAG 补充信息
    rag_response = _rag_query_engine.query(f"提供关于 '{entity_name}' 的详细信息。")
    rag_context = "\n\n".join([node.get_content() for node in rag_response.source_nodes])
    reasoning_path.append(f"步骤 3: 通过 RAG 检索关于 '{entity_name}' 的背景文档信息")

    # 4. LLM 生成最终回答
    final_answer_prompt = PromptTemplate(
        "你是一个专业的金融分析师。请根据以下信息，以清晰、简洁的语言回答用户的问题。\n"
        "--- 用户问题 ---\n{question}\n\n"
        "--- 知识图谱查询结果 ---\n{kg_result}\n\n"
        "--- 相关文档信息 ---\n{rag_context}\n\n"
        "--- 最终回答 ---\n"
    )

    formatted_prompt = final_answer_prompt.format(
        question=question,
        kg_result=kg_result_text,
        rag_context=rag_context
    )

    reasoning_path.append("步骤 4: 综合图谱结果和文档信息，由 LLM 生成最终回答")
    final_response = Settings.llm.complete(formatted_prompt)
    final_answer = final_response.text

    return {
        "final_answer": final_answer,
        "reasoning_path": reasoning_path
    }


# ============= FastAPI 应用 =============
app = FastAPI(
    title="GraphRAG 多跳问答系统",
    description="融合文档检索、图谱推理的多跳问答系统",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    _initialize_settings()
    _initialize_rag_index()
    _initialize_kg_query_engine()
    print("[INFO] GraphRAG 系统启动完成")


@app.get("/")
def read_root():
    return {
        "message": "欢迎使用 GraphRAG API",
        "instructions": "请先运行图谱构建脚本构建知识图谱，然后访问 /docs 查看 API 文档"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "graph-rag"}


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", example="星辰科技的最大股东是谁？")


class QueryResponse(BaseModel):
    final_answer: str
    reasoning_path: List[str]


@app.post("/query", response_model=QueryResponse)
async def query_graph_rag(request: QueryRequest):
    """执行 GraphRAG 查询"""
    try:
        result = multi_hop_query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= 命令行入口 =============
def main():
    """命令行入口"""
    print("=" * 60)
    print("GraphRAG 多跳问答系统")
    print("=" * 60)

    # 初始化
    _initialize_settings()
    _initialize_rag_index()
    _initialize_kg_query_engine()

    if not _graph_store:
        raise RuntimeError("图谱存储未初始化")

    # 测试查询
    test_questions = [
        "星辰科技的最大股东是谁？",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}")

        try:
            result = multi_hop_query(q)
            print(f"\n回答:\n{result['final_answer']}\n")
            print(f"\n推理路径:")
            for step in result['reasoning_path']:
                print(f"  {step}")
        except Exception as e:
            print(f"查询失败: {e}")

    print("\n" + "=" * 60)
    print("\n启动 API 服务器...")
    print(f"访问 http://{_config.API_HOST}:{_config.API_PORT}/docs 查看 API 文档")
    uvicorn.run(app, host=_config.API_HOST, port=_config.API_PORT)


if __name__ == "__main__":
    main()
