"""
作业二: 为 LlamaIndex 构建 OCR 图像文本加载器
基于 PaddleOCR 的多模态数据接入
"""
import os
from typing import List, Union, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

from llama_index.core.readers.base import BaseReader
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

# PaddleOCR 导入
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("请先安装 PaddleOCR: pip install 'paddleocr<3.0'")
    raise

# 加载环境变量
load_dotenv()


class ImageOCRReader(BaseReader):
    """
    使用 PP-OCR v5 从图像中提取文本并返回 Document

    这是一个自定义的 LlamaIndex Reader，能够从图像文件中提取文本内容，
    并将其转换为可用于索引和检索的 Document 对象。
    """

    def __init__(
        self,
        lang: str = 'ch',
        use_gpu: bool = False,
        ocr_version: str = "PP-OCRv4",
        **kwargs
    ):
        """
        初始化 ImageOCRReader

        Args:
            lang: OCR 语言 ('ch' 中文, 'en' 英文, 'fr' 法文等)
            use_gpu: 是否使用 GPU 加速
            ocr_version: OCR 版本 ('PP-OCRv3', 'PP-OCRv4', 'PP-OCRv5')
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_version = ocr_version

        # 初始化 PaddleOCR
        device = "gpu" if use_gpu else "cpu"
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # 使用方向分类器
            lang=lang,
            use_gpu=use_gpu,
            device=device,
            ocr_version=ocr_version,
            show_log=False,
            **kwargs
        )

    def load_data(self, file: Union[str, List[str]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表

        Args:
            file: 图像路径字符串 或 路径列表

        Returns:
            List[Document]: 包含提取文本的 Document 对象列表
        """
        if isinstance(file, str):
            files = [file]
        else:
            files = file

        documents = []
        for img_path in files:
            doc = self._process_single_image(img_path)
            if doc:
                documents.append(doc)

        return documents

    def _process_single_image(self, img_path: str) -> Document:
        """
        处理单个图像文件

        Args:
            img_path: 图像文件路径

        Returns:
            Document: 包含 OCR 结果的 Document 对象
        """
        if not os.path.exists(img_path):
            print(f"警告: 文件不存在 - {img_path}")
            return None

        # 执行 OCR
        result = self.ocr.ocr(img_path, cls=True)

        if not result or not result[0]:
            print(f"警告: 未能从图像中提取文本 - {img_path}")
            return Document(text="", metadata={"image_path": img_path, "error": "No text detected"})

        # 解析 OCR 结果
        text_blocks = []
        total_confidence = 0
        block_count = 0

        for line in result[0]:
            if line:
                # line 格式: [[bbox], (text, confidence)]
                bbox = line[0]
                text_info = line[1]
                text = text_info[0]
                confidence = text_info[1]

                text_blocks.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox
                })
                total_confidence += confidence
                block_count += 1

        # 计算平均置信度
        avg_confidence = total_confidence / block_count if block_count > 0 else 0

        # 构建文本内容
        # 方式1: 简单拼接所有文本
        simple_text = "\n".join([block["text"] for block in text_blocks])

        # 方式2: 带置信度的文本
        annotated_text = "\n".join([
            f"{block['text']} (conf: {block['confidence']:.2f})"
            for block in text_blocks
        ])

        # 构建最终文本（使用简单拼接）
        final_text = simple_text

        # 构建元数据
        metadata = {
            "image_path": img_path,
            "image_name": os.path.basename(img_path),
            "ocr_model": f"PP-OCR-{self.ocr_version}",
            "language": self.lang,
            "num_text_blocks": block_count,
            "avg_confidence": round(avg_confidence, 4),
            "file_size": os.path.getsize(img_path) if os.path.exists(img_path) else 0,
        }

        return Document(text=final_text, metadata=metadata)

    def load_data_from_dir(self, dir_path: str, extensions: Union[str, List[str]] = None) -> List[Document]:
        """
        批量处理目录中的所有图像文件

        Args:
            dir_path: 目录路径
            extensions: 文件扩展名列表，如 ['.jpg', '.png', '.jpeg']
                       如果为 None，则处理所有常见图像格式

        Returns:
            List[Document]: 包含所有图像 OCR 结果的 Document 列表
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        elif isinstance(extensions, str):
            extensions = [extensions]

        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"目录不存在或不是有效目录: {dir_path}")

        image_files = []
        for ext in extensions:
            image_files.extend(dir_path.glob(f"*{ext}"))
            image_files.extend(dir_path.glob(f"*{ext.upper()}"))

        # 去重并排序
        image_files = sorted(set(image_files))

        print(f"在目录 {dir_path} 中找到 {len(image_files)} 个图像文件")

        return self.load_data([str(f) for f in image_files])

    def visualize_ocr_results(self, img_path: str, output_path: str = None):
        """
        可视化 OCR 检测框（需要 OpenCV）

        Args:
            img_path: 输入图像路径
            output_path: 输出图像路径，如果为 None 则不保存
        """
        try:
            import cv2
        except ImportError:
            print("需要安装 OpenCV: pip install opencv-python")
            return

        # 执行 OCR
        result = self.ocr.ocr(img_path, cls=True)

        if not result or not result[0]:
            print(f"没有检测到文本: {img_path}")
            return

        # 读取图像
        image = cv2.imread(img_path)

        # 绘制检测框
        for line in result[0]:
            if line:
                bbox = line[0]
                bbox = [[int(p[0]), int(p[1])] for p in bbox]
                cv2.polylines(image, [bbox], True, (0, 255, 0), 2)

        # 保存或显示
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"可视化结果已保存到: {output_path}")
        else:
            # 可以选择显示图像
            print("可视化完成（未保存）")


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


def test_ocr_reader():
    """测试 ImageOCRReader"""
    print("\n" + "="*80)
    print("OCR 图像加载器测试")
    print("="*80)

    # 初始化 reader
    reader = ImageOCRReader(lang='ch', use_gpu=False)

    # 测试图像目录
    image_dir = Path(__file__).parent.parent / "data" / "ocr_images"

    if not image_dir.exists():
        print(f"警告: 图像目录不存在 - {image_dir}")
        print("请准备测试图像并放在 data/ocr_images/ 目录下")
        return

    # 加载图像
    print(f"\n从目录加载图像: {image_dir}")
    documents = reader.load_data_from_dir(str(image_dir))

    print(f"\n成功处理 {len(documents)} 个图像文件")

    # 打印每个文档的信息
    for i, doc in enumerate(documents):
        print(f"\n{'='*60}")
        print(f"图像 {i+1}: {doc.metadata.get('image_name', 'Unknown')}")
        print(f"{'='*60}")
        print(f"检测到文本块数: {doc.metadata.get('num_text_blocks', 0)}")
        print(f"平均置信度: {doc.metadata.get('avg_confidence', 0):.4f}")
        print(f"OCR 模型: {doc.metadata.get('ocr_model', 'Unknown')}")
        print(f"\n提取的文本:")
        print("-" * 40)
        print(doc.text[:500] + "..." if len(doc.text) > 500 else doc.text)

    return documents


def test_llamaindex_integration(documents: List[Document]):
    """测试与 LlamaIndex 的集成"""
    print("\n" + "="*80)
    print("LlamaIndex 集成测试")
    print("="*80)

    # 设置模型
    setup_deepseek_model()

    # 构建索引
    print("\n构建向量索引...")
    index = VectorStoreIndex.from_documents(documents)
    print("索引构建完成!")

    # 创建查询引擎
    query_engine = index.as_query_engine(similarity_top_k=2, streaming=False)

    # 测试查询
    test_queries = [
        "图片中提到了什么内容？",
        "有哪些文字信息？",
        "文档的主题是什么？"
    ]

    print("\n执行测试查询:")
    for query in test_queries:
        print(f"\n查询: {query}")
        response = query_engine.query(query)
        print(f"回答: {response}")


def main():
    """主函数"""
    print("开始运行 OCR 图像加载器实验...")

    # 检查环境变量（可选，如果使用 OpenAI 进行问答）
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未找到 OPENAI_API_KEY 环境变量")
        print("OCR 功能不需要 API KEY，但集成测试需要")
        print("请在 .env 文件中设置: OPENAI_API_KEY=your_key_here")

    # 测试 OCR Reader
    documents = test_ocr_reader()

    if documents and os.getenv("OPENAI_API_KEY"):
        # 测试与 LlamaIndex 的集成
        test_llamaindex_integration(documents)

    print("\n实验完成！请查看上面的输出结果，并参考 report.md 模板撰写实验报告。")


if __name__ == "__main__":
    main()
