
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
import httpx
from typing import Optional

app = FastAPI(title="Ollama FastAPI Proxy", version="1.0.0")
llm_api = APIRouter()

# 定义 Ollama API 的 URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama 生成接口
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"     # Ollama 聊天接口

# 数据模型，用于接收请求数据
class GenerateRequest(BaseModel):
    model: str = "qwen3:8b"  # 模型名称
    prompt: str  # 输入的 prompt
    temperature: float = 0.1  # 温度参数，默认为 0.1
    max_tokens: int = 1000  # 最大生成的 token 数，默认为 1000
    stream: bool = False  # 是否使用流式响应

class ChatRequest(BaseModel):
    model: str = "qwen3:8b"  # 模型名称
    messages: list  # 聊天消息列表
    temperature: float = 0.1  # 温度参数
    stream: bool = False  # 是否使用流式响应

# 符合不同 API 的使用场景和语义
# - GenerateRequest 用于文本生成接口 ( /generate )，使用 prompt 字段和 max_tokens
# - ChatRequest 用于聊天接口 ( /chat )，使用 messages 字段 （消息列表），通常不需要 max_tokens
# - 这样更符合不同 API 的使用场景和语义

# 在扩展性上面，不同接口可能会有不同的参数需求
# - 例如聊天接口可能需要 system_message 、 context_length 等
# - 生成接口可能需要 stop_sequences 、 top_p 等



@llm_api.post('/generate')
async def generate_text(request: GenerateRequest):
    """生成文本接口，支持流式和非流式响应"""
    # 构建请求数据
    data = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": request.stream,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens
        }
    }
    
    if request.stream:
        # 流式响应
        async def generate_stream():
            async with httpx.AsyncClient() as client:
                async with client.stream('POST', OLLAMA_API_URL, json=data) as response:
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'error': f'Ollama API error: {response.status_code}'})}\n\n"
                        return
                    
                    async for chunk in response.aiter_lines():
                        if chunk:
                            try:
                                chunk_data = json.loads(chunk)
                                if chunk_data.get('response'):
                                    yield f"data: {json.dumps({'text': chunk_data['response']})}\n\n"
                                if chunk_data.get('done', False):
                                    yield f"data: {json.dumps({'done': True})}\n\n"
                                    break
                                elif chunk.strip() == '':
                                    continue
                            except json.JSONDecodeError:
                                # 如果不是JSON格式，直接输出原始内容
                                if chunk.strip():
                                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                                continue
        
        return StreamingResponse(generate_stream(), media_type="text/plain")
    else:
        # 非流式响应
        async with httpx.AsyncClient() as client:
            response = await client.post(OLLAMA_API_URL, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "generated_text": result.get("response", "")
                }
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)

@llm_api.post('/chat')
async def chat(request: ChatRequest):
    """聊天接口，支持流式和非流式响应"""
    # 构建请求数据
    data = {
        "model": request.model,
        "messages": request.messages,
        "stream": request.stream,
        "options": {
            "temperature": request.temperature
        }
    }
    
    if request.stream:
        # 流式响应
        async def chat_stream():
            async with httpx.AsyncClient() as client:
                async with client.stream('POST', OLLAMA_CHAT_URL, json=data) as response:
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'error': f'Ollama API error: {response.status_code}'})}\n\n"
                        return
                    
                    async for chunk in response.aiter_lines():
                        if chunk:
                            try:
                                chunk_data = json.loads(chunk)
                                if chunk_data.get('message', {}).get('content'):
                                    yield f"data: {json.dumps({'content': chunk_data['message']['content']})}\n\n"
                                if chunk_data.get('done', False):
                                    yield f"data: {json.dumps({'done': True})}\n\n"
                                    break
                                elif chunk.strip() == '':
                                    continue
                            except json.JSONDecodeError:
                                # 如果不是JSON格式，直接输出原始内容
                                if chunk.strip():
                                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                                continue
        
        return StreamingResponse(chat_stream(), media_type="text/plain")
    else:
        # 非流式响应
        async with httpx.AsyncClient() as client:
            response = await client.post(OLLAMA_CHAT_URL, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "message": result.get("message", {})
                }
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)

# 注册路由
app.include_router(llm_api, prefix="/api/v1", tags=["LLM"])

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Ollama FastAPI Proxy is running"}

# 根路径
@app.get("/")
async def root():
    return {
        "message": "Welcome to Ollama FastAPI Proxy",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/v1/generate",
            "chat": "/api/v1/chat",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



# windows
# Invoke-RestMethod -Uri "http://localhost:11434/api/chat" -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"model": "qwen3:8b", "messages": [{"role": "user", "content": "你好"}], "stream": false}'

# Linux
# curl -X 'POST' \
#   'http://127.0.0.1:8000/api/v1/chat' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "model": "qwen3:8b",
#   "messages": [
#     {"role": "user", "content": "你好"}
#   ],
#   "temperature": 0.1,
#   "stream": true
# }'