"""
智能客服系统入口
支持多轮对话、工具调用和模型/插件热更新
"""
import uvicorn
from smart_customer_service.api import app


def main():
    """
    启动智能客服 FastAPI 服务器
    """
    print("=" * 50)
    print("🚀 启动智能客服系统...")
    print("=" * 50)
    print("API 文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
