# Celery 集成说明

本项目已集成 Celery 异步任务队列，所有数据库读写操作都通过 Celery 异步处理。

## 架构变更

**原架构**: Python FastAPI → 数据库  
**新架构**: Python FastAPI → Celery → 数据库

## 新增组件

### 1. Redis (消息代理)
- 作为 Celery 的消息代理和结果后端
- 默认端口: 6379

### 2. Celery Worker
- 处理异步数据库任务
- 支持并发处理

### 3. Celery Beat
- 定时任务调度器
- 用于清理过期数据等定时任务

### 4. Flower
- Celery 监控界面
- 访问地址: http://localhost:5555

## 启动方式

### 方式一: 使用 Docker Compose (推荐)
```bash
docker-compose up -d
```

### 方式二: 手动启动服务

1. **启动 Redis**
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

2. **启动 Celery 服务** (Windows)
```bash
start_services.bat
```

3. **启动 FastAPI 应用**
```bash
python main.py
```

### 方式三: 使用 Python 脚本
```bash
python start_celery_services.py
```

## 新增的 API 端点

### `/task/{task_id}`
- 查询 Celery 任务状态
- 返回任务执行结果或状态

示例:
```bash
curl http://localhost:8000/task/your-task-id
```

## 配置说明

### 环境变量
在 `.env` 文件中添加以下配置:

```env
# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# 其他现有配置...
```

### Celery 配置
- 配置文件: `celery_app.py`
- 任务文件: `celery_tasks.py`
- 队列配置: 数据库任务路由到 'database' 队列

## 数据库操作变更

所有数据库操作现在都是异步的:

1. **保存对话历史**: `save_conversation_task.delay()`
2. **获取对话历史**: `get_conversation_history_task.delay()`
3. **删除对话历史**: `delete_conversation_history_task.delay()`
4. **初始化数据库**: `init_database_task.delay()`

## 监控和调试

### Flower 监控界面
- URL: http://localhost:5555
- 功能: 查看任务状态、Worker 状态、队列情况

### 日志
- Celery Worker 日志: 显示任务执行情况
- FastAPI 日志: 显示 API 请求和任务提交情况

## 故障排除

### 1. Redis 连接失败
```bash
# 检查 Redis 是否运行
docker ps | grep redis

# 启动 Redis
docker run -d -p 6379:6379 redis:7-alpine
```

### 2. Celery Worker 无法启动
```bash
# 检查依赖是否安装
pip install -r requirements.txt

# 手动启动 Worker
celery -A celery_app worker --loglevel=info
```

### 3. 任务执行失败
- 检查 Flower 监控界面
- 查看 Worker 日志
- 确认数据库连接正常

## 性能优化

### Worker 并发设置
```bash
# 调整并发数
celery -A celery_app worker --concurrency=8
```

### 任务超时设置
在 `celery_app.py` 中调整:
```python
task_soft_time_limit = 300  # 5分钟软超时
task_time_limit = 600       # 10分钟硬超时
```

## 开发注意事项

1. **异步特性**: 所有数据库操作现在都是异步的，需要等待任务完成
2. **错误处理**: 任务可能失败，需要检查任务状态
3. **超时设置**: 长时间运行的任务需要适当的超时设置
4. **监控**: 使用 Flower 监控任务执行情况

## 测试

运行以下命令测试集成:

```bash
# 测试 API 健康检查
curl http://localhost:8000/health

# 测试对话历史获取
curl http://localhost:8000/history

# 测试工作流运行
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello", "session_id": "test"}'
```