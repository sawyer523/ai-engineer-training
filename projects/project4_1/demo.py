import os
import sqlite3
import json
import time
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
import operator

# 第三方库导入
from pydantic import BaseModel, Field
import dashscope
from dashscope.audio.asr import Recognition
from http import HTTPStatus

# LangChain 导入
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# LangGraph 导入
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# --- 配置 ---
# 说明：实际使用时请在环境变量中设置 DASHSCOPE_API_KEY。
# os.environ["DASHSCOPE_API_KEY"] = "sk-..." 
# 如果未检测到密钥，本演示会使用简易规则引擎（Mock LLM）。

DB_PATH = "orders.db"

# --- 1. 数据库初始化 (SQLite) ---
def setup_database():
    """初始化 SQLite 数据库并写入示例订单数据。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建订单表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        user_id TEXT,
        status TEXT,
        items TEXT,
        logistics_info TEXT,
        created_at TEXT
    )
    ''')
    
    # 检查是否已有数据
    cursor.execute('SELECT count(*) FROM orders')
    if cursor.fetchone()[0] == 0:
        print("正在写入示例订单数据...")
        sample_orders = [
            ("12345", "user_001", "shipped", "Wireless Headphones", "Arrived at Beijing Sorting Center", datetime.now().isoformat()),
            ("67890", "user_001", "pending_payment", "Smart Watch", "Waiting for payment", datetime.now().isoformat()),
            ("11223", "user_002", "delivered", "Laptop Stand", "Delivered to locker", datetime.now().isoformat()),
        ]
        cursor.executemany('INSERT INTO orders VALUES (?,?,?,?,?,?)', sample_orders)
        conn.commit()
    
    # LangGraph 对话检查点持久化说明：
    # SqliteSaver 会自动创建所需的检查点表，本演示直接依赖其默认行为。
    
    conn.close()

# --- 2. RAG 初始化 (知识库) ---
def setup_rag_retriever():
    """初始化一个用于检索政策知识的简易 RAG 检索器。"""
    policies = [
        "退款政策：自签收之日起 7 天内，且商品未拆封，可发起退款申请。",
        "物流政策：满 50 美元免邮，标准配送一般为 3-5 个工作日。",
        "质保政策：电子类商品享受 1 年制造商质保服务。",
        "支付政策：支持信用卡、PayPal 和支付宝。",
        "订单修改：订单状态变为“已发货”后不可再修改订单信息。"
    ]
    
    documents = [Document(page_content=p, metadata={"source": "policy_doc"}) for p in policies]
    
    # 说明：演示环境使用 FakeEmbeddings，无需外部 API。
    # 生产环境可替换为 OpenAIEmbeddings 或 DashScopeEmbeddings。
    embeddings = FakeEmbeddings(size=768) 
    
    vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

# --- 3. 工具定义 ---

@tool
def check_order(order_id: str) -> str:
    """根据订单号查询订单状态与物流信息。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT status, items, logistics_info FROM orders WHERE order_id = ?', (order_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        status, items, logistics = result
        return f"订单 {order_id}（{items}）：当前状态为『{status}』。物流信息：{logistics}。"
    else:
        return f"未查询到订单 {order_id}，请检查订单号是否正确。"

@tool
def search_policy(query: str) -> str:
    """查询与客服政策相关的知识（退款、物流等）。"""
    # 说明：真实环境可复用全局检索器；为保证工具无状态，这里简单重新初始化。
    retriever = setup_rag_retriever()
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

# --- 4. 输入处理 (ASR/OCR) ---

def process_audio_input(file_path: str) -> str:
    """使用 Qwen/Dashscope 进行语音转写（演示中为模拟）。"""
    print(f"[系统] 正在处理音频文件：{file_path}")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[系统] 未检测到 API Key，返回模拟 ASR 结果。")
        return "查订单 12345" # 模拟结果
    
    try:
        # 真实实现：使用 DashScope ASR
        # task = dashscope.audio.asr.Recognition.call(...)
        # 演示保持为模拟，返回固定文本。
        return "查订单 12345"
    except Exception as e:
        return f"Error in ASR: {str(e)}"

def process_image_input(file_path: str) -> str:
    """使用 Qwen-VL 进行 OCR（演示中为模拟）。"""
    print(f"[系统] 正在处理图片文件：{file_path}")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[系统] 未检测到 API Key，返回模拟 OCR 结果。")
        return "图片中订单号似乎是 67890" # 模拟结果

    try:
        # 真实实现：使用 DashScope 多模态
        # messages = [{...}]
        # response = dashscope.MultiModalConversation.call(model='qwen-vl-max', messages=messages)
        return "图片中订单号似乎是 67890"
    except Exception as e:
        return f"Error in OCR: {str(e)}"

# --- 5. LangGraph 状态与节点 ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    order_id: Optional[str]

# LLM 初始化
# 如果存在密钥则使用真实模型，否则使用简易规则引擎（Mock）。
api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
if api_key:
    # 使用 ChatOpenAI（如需对接 Qwen 可设置兼容的 base_url）；此处默认使用 OpenAI。
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
else:
    # 未检测到密钥，使用规则引擎模拟。
    print("[系统] 未检测到 LLM API Key，使用规则引擎（Mock）进行演示。")
    class MockLLM:
        def invoke(self, messages):
            last_msg = messages[-1].content.lower()
            if "查订单" in last_msg or "订单" in last_msg or "12345" in last_msg or "67890" in last_msg:
                # 需要调用工具时返回 tool_calls
                if "12345" in last_msg:
                    return AIMessage(content="", tool_calls=[{"name": "check_order", "args": {"order_id": "12345"}, "id": "call_1"}])
                elif "67890" in last_msg:
                    return AIMessage(content="", tool_calls=[{"name": "check_order", "args": {"order_id": "67890"}, "id": "call_2"}])
                else:
                    return AIMessage(content="请提供订单号。")
            elif "政策" in last_msg or "退款" in last_msg:
                 return AIMessage(content="", tool_calls=[{"name": "search_policy", "args": {"query": last_msg}, "id": "call_3"}])
            else:
                return AIMessage(content="我可以帮您查询订单或解答政策相关问题。")
        
        def bind_tools(self, tools):
            return self

    llm = MockLLM()

# 绑定工具
tools = [check_order, search_policy]
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: AgentState):
    """根据历史消息决定要采取的动作。"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def input_processing_node(state: AgentState):
    """对输入进行预处理（文本/音频/图片）。"""
    # 说明：真实系统可拆分更细；此处假定用户消息已加入状态，若是文件路径则进行相应处理。
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        content = last_message.content
        if content.endswith(".wav") or content.endswith(".mp3"):
            text = process_audio_input(content)
            return {"messages": [HumanMessage(content=f"音频转写：{text}")]}
        elif content.endswith(".jpg") or content.endswith(".png"):
            text = process_image_input(content)
            return {"messages": [HumanMessage(content=f"图片识别：{text}")]}
    return {"messages": []} # Return empty update explicitly to avoid InvalidUpdateError if that's the cause

# 定义图工作流
workflow = StateGraph(AgentState)
workflow.add_node("input_proc", input_processing_node)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "input_proc")
workflow.add_edge("input_proc", "chatbot")

def route_tools(state: AgentState):
    """条件路由：判断是否需要调用工具。"""
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get("messages", [])):
        ai_message = messages[-1]
    elif isinstance(state, BaseMessage):
        ai_message = state
    else:
        raise ValueError(f"未在状态中找到消息，无法进行工具路由：{state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

workflow.add_conditional_edges("chatbot", route_tools)
workflow.add_edge("tools", "chatbot")

# 配置检查点持久化
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

app = workflow.compile(checkpointer=memory)

# --- 6. 主流程执行与测试 ---

def run_demo():
    print("--- 订单查询客服演示 ---")
    setup_database()
    
    # 对话会话 ID（用于状态持久化）
    config = {"configurable": {"thread_id": "session_1"}}
    
    # 模拟用户输入场景
    scenarios = [
        "你好，我要查订单。",
        "订单号是 12345。",
        "如果不喜欢可以退货吗？", # RAG 检索
        "audio_sample.wav", # 模拟音频
        "order_image.jpg"   # 模拟图片
    ]
    
    for user_input in scenarios:
        print(f"\n用户：{user_input}")
        
        # 判断是文件路径还是文本
        msg_content = user_input
        
        # 以流式方式运行图
        events = app.stream(
            {"messages": [HumanMessage(content=msg_content)]}, 
            config, 
            stream_mode="values"
        )
        
        for event in events:
            if "messages" in event:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    print(f"客服：{last_msg.content}")
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                         print(f"  [工具调用]：{last_msg.tool_calls[0]['name']}")

if __name__ == "__main__":
    run_demo()
