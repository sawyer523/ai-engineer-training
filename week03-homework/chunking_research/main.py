"""
作业一: 探索 LlamaIndex 中的句子切片检索及其参数影响分析
使用 DeepSeek 模型
"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SentenceWindowNodeParser,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

# 加载环境变量
load_dotenv()


def setup_deepseek_model():
    """配置 OpenAI 模型和嵌入模型（使用 api.apiyi.com 代理）"""
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://api.apiyi.com/v1"
    )

    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://api.apiyi.com/v1",
        embed_batch_size=6
    )


def load_sample_documents() -> List[Document]:
    """从 data 目录加载示例文档数据集"""
    from llama_index.core import SimpleDirectoryReader
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data"

    # 使用 SimpleDirectoryReader 加载所有 .txt 文件
    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        required_exts=[".txt"],
        recursive=False
    )

    documents = reader.load_data()
    print(f"从 {data_dir} 加载了 {len(documents)} 篇文档:")
    for doc in documents:
        print(f"  - {doc.metadata.get('file_name', 'Unknown')}")

    return documents


def evaluate_splitter(splitter, documents: List[Document], query: str, splitter_name: str) -> Dict[str, Any]:
    """
    评估指定切片器的效果

    Args:
        splitter: 切片器实例
        documents: 文档列表
        query: 测试查询
        splitter_name: 切片器名称

    Returns:
        包含评估结果的字典
    """
    print(f"\n{'='*60}")
    print(f"评估切片器: {splitter_name}")
    print(f"{'='*60}")

    # 构建索引
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter] if splitter not in [None, "none"] else None
    )

    # 创建查询引擎
    if isinstance(splitter, SentenceWindowNodeParser):
        # 句子窗口切片需要特殊后处理器
        query_engine = index.as_query_engine(
            similarity_top_k=2,
            node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
        )
    else:
        query_engine = index.as_query_engine(
            similarity_top_k=2,
            streaming=False
        )

    # 执行查询
    response = query_engine.query(query)

    # 获取检索到的节点
    source_nodes = response.source_nodes

    # 打印结果
    print(f"\n查询: {query}")
    print(f"\n回答: {response.response}")
    print(f"\n检索到 {len(source_nodes)} 个相关节点:")

    for i, node in enumerate(source_nodes):
        print(f"\n--- 节点 {i+1} ---")
        print(f"内容片段: {node.get_content()[:200]}...")
        print(f"相似度得分: {node.score:.4f}")

    return {
        "splitter_name": splitter_name,
        "response": str(response),
        "num_nodes": len(source_nodes),
        "source_nodes": source_nodes
    }


def run_comparison_experiments():
    """运行切片方式对比实验"""
    print("\n" + "="*80)
    print("句子切片检索对比实验")
    print("="*80)

    # 设置模型
    setup_deepseek_model()

    # 加载文档
    documents = load_sample_documents()
    print(f"\n加载了 {len(documents)} 篇文档")

    # 定义测试查询
    queries = [
        "什么是量子纠缠？",
        "气候变化的主要原因是什么？",
        "Transformer架构有什么作用？"
    ]

    # 定义不同的切片方式
    splitters = {
        "Sentence Splitter (chunk_size=512, overlap=50)": SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        ),
        "Sentence Splitter (chunk_size=256, overlap=25)": SentenceSplitter(
            chunk_size=256,
            chunk_overlap=25
        ),
        "Sentence Splitter (chunk_size=128, overlap=0)": SentenceSplitter(
            chunk_size=128,
            chunk_overlap=0
        ),
        "Token Splitter": TokenTextSplitter(
            chunk_size=128,
            chunk_overlap=16,
            separator="\n"
        ),
        "Sentence Window (window_size=1)": SentenceWindowNodeParser.from_defaults(
            window_size=1,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        ),
        "Sentence Window (window_size=3)": SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        ),
        "Sentence Window (window_size=5)": SentenceWindowNodeParser.from_defaults(
            window_size=5,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        ),
    }

    # 运行实验
    results = []

    for splitter_name, splitter in splitters.items():
        for query in queries:
            result = evaluate_splitter(splitter, documents, query, splitter_name)
            result["query"] = query
            results.append(result)

    # 打印总结
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)

    for splitter_name in splitters.keys():
        splitter_results = [r for r in results if r["splitter_name"] == splitter_name]
        avg_nodes = sum(r["num_nodes"] for r in splitter_results) / len(splitter_results)
        print(f"\n{splitter_name}:")
        print(f"  平均检索节点数: {avg_nodes:.1f}")

    return results


def main():
    """主函数"""
    print("开始运行句子切片检索实验...")

    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("错误: 未找到 OPENAI_API_KEY 环境变量")
        print("请在 .env 文件中设置: OPENAI_API_KEY=your_key_here")
        return

    # 运行对比实验
    results = run_comparison_experiments()

    print("\n实验完成！请查看上面的输出结果，并参考 report.md 模板撰写实验报告。")


if __name__ == "__main__":
    main()
