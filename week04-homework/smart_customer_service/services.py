import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from .tools import default_tools


load_dotenv()


class ServiceManager:
    """
    一个用于管理和提供模型及工具的服务管理器。
    """
    def __init__(self):
        print("正在初始化 LLM 和工具...")
        self._llm = self._create_llm()
        self._tools = default_tools
        print("✅ ServiceManager 初始化完成。")
        self.print_services()

    def _create_llm(self):
        if not os.environ.get("DASHSCOPE_API_KEY"):
            print("⚠️ 警告: DASHSCOPE_API_KEY 环境变量未设置！")
        return ChatTongyi(
            model_name="qwen-plus",
            temperature=0,
            streaming=True
        )

    def get_llm(self):
        return self._llm

    def get_tools(self) -> list:
        return self._tools

    def update_llm(self, model_name: str):
        print(f"🔄 [热更新] 正在更新LLM模型为: {model_name}")
        self._llm = ChatTongyi(model_name=model_name, temperature=0, streaming=True)
        self.print_services()

    def update_tools(self, new_tools: list):
        print("🔄 [热更新] 正在更新工具列表...")
        self._tools = new_tools
        self.print_services()

    def print_services(self):
        print("--- 当前服务状态 ---")
        print(f"  模型: {self._llm.model_name}")
        print(f"  工具: {[tool.name for tool in self._tools]}")
        print("--------------------")

    def get_services_status(self) -> dict:
        return {
            "model": self._llm.model_name,
            "tools": [tool.name for tool in self._tools]
        }


# 创建一个单例
service_manager = ServiceManager()
