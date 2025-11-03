# WebLLM Docker 部署指南

本项目支持使用Docker进行一键部署，包含所有必要的服务组件。

## 🏗️ 架构说明

Docker部署包含以下服务：
- **app**: 主应用服务（FastAPI + Gradio）
- **postgres**: PostgreSQL数据库
- **redis**: Redis缓存服务

## 🚀 快速开始

### 1. 环境准备

确保已安装：
- Docker
- Docker Compose

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.docker .env

# 编辑.env文件，填入你的配置
# 特别注意设置DASHSCOPE_API_KEY
```

### 3. 启动服务

**Windows用户：**
```cmd
docker-start.bat
```

**Linux/Mac用户：**
```bash
chmod +x docker-start.sh
./docker-start.sh
```

**或者手动启动：**
```bash
docker-compose up --build -d
```

### 4. 访问服务

- **Gradio界面**: http://localhost:7860
- **FastAPI文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 📊 服务管理

### 查看服务状态
```bash
docker-compose ps
```

### 查看日志
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f app
docker-compose logs -f postgres
docker-compose logs -f redis
```

### 停止服务
```bash
docker-compose down
```

### 重启服务
```bash
docker-compose restart
```

### 清理数据（谨慎使用）
```bash
# 停止服务并删除数据卷
docker-compose down -v
```

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云API密钥 | 必填 |
| `DATABASE_NAME` | 数据库名称 | webllm_db |
| `DATABASE_USER` | 数据库用户 | webllm_user |
| `DATABASE_PASSWORD` | 数据库密码 | webllm_password |
| `LLM_MODEL` | LLM模型 | qwen-turbo |
| `LOG_LEVEL` | 日志级别 | INFO |

### 端口映射

| 服务 | 容器端口 | 主机端口 |
|------|----------|----------|
| 应用 | 8000 | 8000 |
| Gradio | 7860 | 7860 |
| PostgreSQL | 5432 | 5432 |
| Redis | 6379 | 6379 |

## 🐛 故障排除

### 1. 服务启动失败

检查日志：
```bash
docker-compose logs app
```

常见问题：
- API密钥未设置或无效
- 端口被占用
- 磁盘空间不足

### 2. 数据库连接失败

检查PostgreSQL服务：
```bash
docker-compose logs postgres
```

### 3. Redis连接失败

检查Redis服务：
```bash
docker-compose logs redis
```

### 4. 重置环境

完全重置（会删除所有数据）：
```bash
docker-compose down -v
docker system prune -f
docker-compose up --build -d
```

## 📈 性能优化

### 1. 资源限制

在docker-compose.yml中添加资源限制：
```yaml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### 2. 数据持久化

数据卷自动持久化：
- `postgres_data`: PostgreSQL数据
- `redis_data`: Redis数据

### 3. 网络优化

服务间通信使用内部网络，提高性能和安全性。

## 🔒 安全建议

1. **环境变量安全**：
   - 不要将.env文件提交到版本控制
   - 使用强密码
   - 定期更换API密钥

2. **网络安全**：
   - 生产环境中限制端口访问
   - 使用反向代理（如Nginx）
   - 启用HTTPS

3. **数据安全**：
   - 定期备份数据库
   - 监控日志异常

## 📝 开发模式

开发时可以挂载代码目录：
```yaml
volumes:
  - .:/app
  - /app/__pycache__  # 排除缓存目录
```

这样可以实时看到代码变更效果。