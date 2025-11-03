FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 配置pip镜像源并安装Python依赖
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip install --no-cache-dir --timeout 300 -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量（不包含敏感数据）
ENV HOST=0.0.0.0
ENV PORT=8000
ENV GRADIO_HOST=0.0.0.0
ENV GRADIO_PORT=7860
ENV LLM_MODEL=qwen-turbo
ENV MAX_INPUT_LENGTH=500
ENV MIN_INPUT_LENGTH=1
ENV RETRY_MULTIPLIER=1
ENV RETRY_MIN_WAIT=2
ENV RETRY_MAX_WAIT=10
ENV RETRY_MAX_ATTEMPTS=3
ENV LOG_LEVEL=INFO

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000 7860

# 启动应用（使用start_all.py来同时启动FastAPI和Gradio）
CMD ["python", "start_all.py"]