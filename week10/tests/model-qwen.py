from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import asyncio
import logging
import dotenv
import os

# 设置日志级别为DEBUG
logging.basicConfig(level=logging.INFO)

# 加载环境变量
dotenv.load_dotenv()

# embedding
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

@tool
async def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    text = f"It's always sunny in {city}!"
    logging.debug(text)
    return text

async def main():
    logging.basicConfig(level=logging.DEBUG)
    agent = create_agent(
        model=ChatTongyi(model="qwen-turbo"),
        tools=[get_weather],
        checkpointer=InMemorySaver(),
        system_prompt="你是一个AI助手，你可以回答用户的问题",
    )

    messages = {"messages": [{"role": "user", "content": "上海的天气怎么样？"}]}
    config = {"configurable": {"thread_id": "1"}}

    result = await agent.ainvoke(messages, config)

    logging.debug(result)

    msgs = result.get("messages", [])
    final_text = ""
    for m in reversed(msgs):
        c = getattr(m, "content", None)
        if isinstance(c, str) and c.strip():
            final_text = c
            break
    print(final_text)

if __name__ == "__main__":
    asyncio.run(main())
