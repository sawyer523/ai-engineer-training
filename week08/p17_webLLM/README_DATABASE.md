# LangGraph AI 对话系统 - 数据库版本

这是一个增强版的LangGraph AI对话系统，新增了PostgreSQL数据库支持和Gradio Web界面。

## 🆕 新功能

### 1. PostgreSQL数据库支持
- 自动保存所有对话历史到数据库
- 支持会话管理和历史查询
- 数据持久化存储

### 2. Gradio Web界面
- 友好的Web对话界面
- 实时对话功能
- 对话历史查询和展示
- 会话管理功能

### 3. 增强的API端点
- `/run` - 处理对话（支持会话ID）
- `/history` - 查询对话历史
- `/health` - 健康检查

## 📋 系统要求

### 软件依赖
- Python 3.8+
- PostgreSQL 12+

### Python包依赖
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 数据库设置

#### 安装PostgreSQL
```bash
# Windows (使用Chocolatey)
choco install postgresql

# 或下载安装包
# https://www.postgresql.org/download/windows/
```

#### 创建数据库
```sql
-- 连接到PostgreSQL
psql -U postgres

-- 创建数据库
CREATE DATABASE langraph_db;

-- 创建用户（可选）
CREATE USER langraph_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE langraph_db TO langraph_user;
```

### 2. 环境配置

复制并配置环境变量：
```bash
cp .env.example .env
```

编辑 `.env` 文件：
```env
# 数据库配置
DATABASE_URL=postgresql://postgres:password@localhost:5432/langraph_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=langraph_db
DB_USER=postgres
DB_PASSWORD=your_password

# API密钥
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 启动服务

#### 方式一：一键启动（推荐）
```bash
python start_all.py
```

#### 方式二：分别启动
```bash
# 终端1：启动API服务器
python main.py

# 终端2：启动Gradio界面
python gradio_app.py
```

### 4. 访问界面

- **Gradio Web界面**: http://localhost:7860
- **FastAPI文档**: http://localhost:8000/docs
- **API健康检查**: http://localhost:8000/health

## 🔧 使用说明

### Web界面使用

1. **AI对话标签页**
   - 在输入框中输入问题
   - 点击"发送"或按Enter键
   - 查看AI回复
   - 使用"新建会话"开始新的对话

2. **历史记录标签页**
   - 查看所有对话历史
   - 按会话ID过滤
   - 设置显示记录数量
   - 实时刷新数据

### API使用

#### 发送对话请求
```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "什么是人工智能？",
    "session_id": "optional-session-id"
  }'
```

#### 查询对话历史
```bash
# 获取最近50条记录
curl "http://localhost:8000/history?limit=50"

# 按会话ID过滤
curl "http://localhost:8000/history?session_id=your-session-id&limit=20"
```

## 📊 数据库结构

### conversation_history 表
```sql
CREATE TABLE conversation_history (
    id SERIAL PRIMARY KEY,
    user_input TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(255)
);
```

### 索引
- `idx_conversation_timestamp` - 按时间排序
- `idx_conversation_session` - 按会话ID查询

## 🛠️ 配置选项

### 环境变量说明

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DATABASE_URL` | 完整数据库连接URL | - |
| `DB_HOST` | 数据库主机 | localhost |
| `DB_PORT` | 数据库端口 | 5432 |
| `DB_NAME` | 数据库名称 | langraph_db |
| `DB_USER` | 数据库用户 | postgres |
| `DB_PASSWORD` | 数据库密码 | password |
| `HOST` | API服务器主机 | 0.0.0.0 |
| `PORT` | API服务器端口 | 8000 |
| `DASHSCOPE_API_KEY` | 通义千问API密钥 | - |

## 🔍 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查PostgreSQL服务是否运行
   - 验证数据库配置信息
   - 确认数据库和用户权限

2. **API调用失败**
   - 检查DASHSCOPE_API_KEY是否正确
   - 验证网络连接
   - 查看日志文件

3. **Gradio界面无法访问**
   - 确认FastAPI服务器已启动
   - 检查端口是否被占用
   - 验证防火墙设置

### 日志查看
```bash
# 查看应用日志
tail -f logs/app.log

# 或在代码中设置日志级别
LOG_LEVEL=DEBUG
```

## 📝 开发说明

### 项目结构
```
p17_webLLM/
├── main.py              # FastAPI应用主文件
├── workflow.py          # LangGraph工作流
├── config.py           # 配置管理
├── database.py         # 数据库模型和管理
├── gradio_app.py       # Gradio Web界面
├── start_all.py        # 一键启动脚本
├── requirements.txt    # Python依赖
├── .env.example       # 环境变量模板
└── README_DATABASE.md # 本文档
```

### 扩展功能
- 添加用户认证
- 实现对话分类和标签
- 增加数据导出功能
- 添加对话统计分析

## 📄 许可证

本项目遵循原项目的许可证条款。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！