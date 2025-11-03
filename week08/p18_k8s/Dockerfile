# Dockerfile
FROM docker.xuanyuan.me/library/python:3.11-slim

# 设置环境变量以支持UTF-8编码
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

WORKDIR /app

# 先复制requirements文件以利用Docker缓存层
COPY requirements.txt .

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

COPY . .

# 暴露FastAPI默认端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]