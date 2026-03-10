"""多代理文章写作系统 - 主入口文件"""


import os
import sys
import datetime
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from .langgraph_app.graph import create_graph


# MCP客户端配置
client = MultiServerMCPClient(
    {
        "tools_server": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
)


async def run_writing_task():
    """连接到MCP服务并执行完整的文章写作任务。"""
    load_dotenv()
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误：请在 .env 文件中设置 DASHSCOPE_API_KEY。")
        return

    # 使用 async with 创建一个到服务器的会话
    async with client.session("tools_server") as mcp_session:
        print("✅ MCP 客户端已连接到工具服务器。")

        app_graph = await create_graph(mcp_session)

        topic = input("请输入您想写的文章主题 (或按回车使用默认主题): ")
        if not topic:
            topic = "帮我写一篇关于AI Agent的文章"

        print("\n" + "=" * 50)
        print("🚀 LangGraph 客户端启动，开始执行写作任务...")
        print("=" * 50 + "\n")

        initial_state = {
            "topic": topic,
            "style": "通俗易懂",
            "length": 1000,
            "research_report": "",
            "draft": "",
            "review_suggestions": "",
            "final_article": "",
            "log": [f"# 多代理协作写作流程记录\n\n**任务主题:** {topic}\n"],
            "error_count": 0,
            "max_retries": 2,
        }

        final_state = await app_graph.ainvoke(initial_state)

        print("\n" + "=" * 50)
        print("✅ 任务完成！")
        print("=" * 50 + "\n")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"article_output_{timestamp}.md"

        final_article = final_state.get("final_article", "未能生成最终文章。")
        process_log = "\n".join(final_state.get("log", []))

        final_output = f"# 最终文章：{topic}\n\n{final_article}\n\n---\n\n{process_log}"

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"🎉 成功生成输出文件：{output_filename}")


def main():
    """
    作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
    在根目录可以通过python -m multi-agent.main 运行
    """
    try:
        asyncio.run(run_writing_task())
    except KeyboardInterrupt:
        print("\n程序已由用户中断。")
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("\n请确保 MCP 服务器正在另一个终端中运行。")
        print("运行命令: python -m multi-agent.mcp_server.main")


if __name__ == "__main__":
    main()
