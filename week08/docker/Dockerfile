FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

# 设置环境变量（不包含敏感数据）
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LLM_MODEL=qwen-turbo
ENV MAX_INPUT_LENGTH=500
ENV MIN_INPUT_LENGTH=1
ENV RETRY_MULTIPLIER=1
ENV RETRY_MIN_WAIT=2
ENV RETRY_MAX_WAIT=10
ENV RETRY_MAX_ATTEMPTS=3
ENV LOG_LEVEL=INFO

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]