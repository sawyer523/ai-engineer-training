# app/main.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
import os
import json

app = FastAPI()

# 设置千问API密钥 - 从环境变量获取
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope_api_key:
    raise ValueError("DASHSCOPE_API_KEY environment variable is required")

# 使用千问模型
llm = ChatTongyi(model="qwen-turbo", dashscope_api_key=dashscope_api_key)

@app.get("/")
async def root():
    return JSONResponse(
        content={"message": "LangChain DevOps Expert API is running!"},
        media_type="application/json; charset=utf-8"
    )

@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy"},
        media_type="application/json; charset=utf-8"
    )

@app.get("/ask")
async def ask_question(question: str):
    try:
        # 构建提示模板
        template = f"""你是一个DevOps专家，专门回答关于Docker和Kubernetes的问题。
问题: {question}
回答:"""
        
        # 直接使用千问聊天模型
        response = llm.invoke([HumanMessage(content=template)])
        
        # 确保返回正确的 UTF-8 编码响应
        result = {"question": question, "answer": response.content}
        
        return JSONResponse(
            content=result,
            media_type="application/json; charset=utf-8"
        )
    except Exception as e:
        error_result = {"error": f"处理请求时出错: {str(e)}"}
        return JSONResponse(
            content=error_result,
            media_type="application/json; charset=utf-8"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# http://localhost:8000/ask?question=你好