import requests
import json
import time

# API 基础 URL
BASE_URL = "http://localhost:8000/api/v1"

def test_generate_non_stream():
    """测试非流式生成接口"""
    print("=== 测试非流式生成接口 ===")
    
    data = {
        "model": "qwen3:8b",
        "prompt": "请介绍一下人工智能的发展历史",
        "temperature": 0.1,
        "max_tokens": 200,
        "stream": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/generate", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"生成的文本: {result.get('generated_text', '')}")
        else:
            print(f"错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

def test_generate_stream():
    """测试流式生成接口"""
    print("\n=== 测试流式生成接口 ===")
    
    data = {
        "model": "qwen3:8b",
        "prompt": "请写一首关于春天的诗",
        "temperature": 0.1,
        "max_tokens": 200,
        "stream": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/generate", json=data, stream=True)
        if response.status_code == 200:
            print("流式响应:")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            data_json = json.loads(line_str[6:])  # 去掉 'data: ' 前缀
                            if 'text' in data_json:
                                print(data_json['text'], end='', flush=True)
                            elif data_json.get('done'):
                                print("\n[完成]")
                                break
                        except json.JSONDecodeError:
                            print(f"解析错误: {line_str}", end='', flush=True)
                            continue
                    elif line_str.strip():
                        # 如果不是标准格式，直接打印
                        print(line_str, end='', flush=True)
        else:
            print(f"错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

def test_chat_non_stream():
    """测试非流式聊天接口"""
    print("\n=== 测试非流式聊天接口 ===")
    
    data = {
        "model": "qwen3:8b",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下自己"}
        ],
        "temperature": 0.1,
        "stream": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=data)
        if response.status_code == 200:
            result = response.json()
            message = result.get('message', {})
            print(f"回复: {message.get('content', '')}")
        else:
            print(f"错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

def test_chat_stream():
    """测试流式聊天接口"""
    print("\n=== 测试流式聊天接口 ===")
    
    data = {
        "model": "qwen3:8b",
        "messages": [
            {"role": "user", "content": "请解释一下什么是机器学习"}
        ],
        "temperature": 0.1,
        "stream": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=data, stream=True)
        if response.status_code == 200:
            print("流式聊天响应:")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            data_json = json.loads(line_str[6:])  # 去掉 'data: ' 前缀
                            if 'content' in data_json:
                                print(data_json['content'], end='', flush=True)
                            elif data_json.get('done'):
                                print("\n[完成]")
                                break
                        except json.JSONDecodeError:
                            continue
        else:
            print(f"错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

def test_health():
    """测试健康检查接口"""
    print("\n=== 测试健康检查接口 ===")
    
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            result = response.json()
            print(f"健康状态: {result}")
        else:
            print(f"错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    print("开始测试 Ollama FastAPI 代理服务...")
    print("请确保:")
    print("1. Ollama 服务正在运行 (http://localhost:11434)")
    print("2. FastAPI 服务正在运行 (http://localhost:8000)")
    print("3. 已安装 qwen3:8b 模型 (或修改代码中的模型名称)")
    print()
    
    # 测试健康检查
    test_health()
    
    # 等待用户确认
    input("按回车键继续测试 API 接口...")
    
    # 测试各个接口    
    test_chat_non_stream()
    time.sleep(1)
    
    test_chat_stream()
    
    print("\n测试完成！")