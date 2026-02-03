# 1 模型 I/O 模块
from langchain_classic.llms import openai
from langchain_classic.chat_models import ChatOpenAI
from langchain_classic.schema import HumanMessage

chat = ChatOpenAI(model='gpt-5')
response = chat([HumanMessage(content="你好")])
print(response.content)


# 2 RAG

from  langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(model='gpt-5'),
    chain_type="stuff",
    retriever = Retriever(type="memory", memory=MemoryType.MEMORY_VALUE, memory_key="some value")
)
qa_chain.run("问题是???")


# 3 存储 memory
# 读取 写入

#         memory（R）  memory(W)
#         /      \   /        \
# input    ｜      LLM      ｜ output

# 4 工具 tools

from  langchain.tools import Tool

def  search_wiki(query):
    return "搜索结果是"

tool = Tool(
    name = "WikiSearch",
    description = "搜索维基百科页面",
    function = search_wiki
)

# 5 callback
# 回调： 特定操作发生时执行预定处理程序的机制
# 两种实现方式： 构造器回调  请求回调

# 回调地狱

# 现代
# Promise、Future 扁平化
# Async Await  以同步风格写异步代码
# 6 LCEL