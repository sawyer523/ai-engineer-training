# 部署说明文档

## 目标与范围
- 说明如何在开发/测试/生产环境部署 `work` 服务，完成依赖安装、配置、初始化、启动与验证。
- 适用组件：FastAPI 应用、LangGraph 对话引擎、FAISS 向量库、SQLite 订单库、SSE 建议推送、日志与指标。

## 环境要求
- 操作系统：Windows 10/11、Linux（x86_64）或 macOS（ARM/Intel）
- 运行时：Python ≥ 3.11（推荐 3.11/3.12；项目已在 CPython 3.13 编译）
- 依赖工具：`pip`、`virtualenv`、`git`；可选 `Docker`/`docker-compose`
- 端口占用：`8000`（应用），`6379`（Redis，可选）

## 依赖安装

### 方式一：本地 Python 环境
- 创建并激活虚拟环境：
  - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
  - Linux/macOS: `python -m venv .venv && source .venv/bin/activate`
- 安装依赖：
  - `pip install fastapi uvicorn[standard] python-dotenv langgraph langchain-community dashscope faiss-cpu redis gradio mcp-server-fastmcp`
  - 如遇 `faiss-cpu` 安装失败，可使用系统包或跳过（仅影响向量检索）。

### 方式二：Docker Compose（推荐一体化）
- 切换到目录：`d:\AI工程化训练营\workspace\week10\work`
- 检查/准备依赖清单：在项目根创建 `requirements.txt`（如无），内容示例：
  ```
  fastapi
  uvicorn[standard]
  python-dotenv
  langgraph
  langchain-community
  dashscope
  faiss-cpu
  redis
  gradio
  mcp-server-fastmcp
  ```
- 启动：`docker-compose up -d`
- 说明：Compose 会安装依赖并以 `uvicorn app:app` 启动；若 `../requirements.txt` 不存在，请按照上方示例创建。

## 配置项
- `.env` 文件（与 `work` 同级，或系统环境变量），关键变量：
  - `MODEL_NAME`：默认 `qwen-turbo`
  - `EMBEDDING_MODEL`：默认 `text-embedding-v4`
  - `KB_INDEX_DIR`：知识库索引目录（默认 `work/faiss_index` 或租户目录）
  - `ORDERS_DB_PATH`：订单数据库路径（默认 `work/db/orders.sqlite` 或租户目录）
  - `CHECKPOINT_DB_PATH`：LangGraph 检查点数据库（可选）
  - `SUPPORT_DB_PATH`：未命中问题记录库（默认 `work/support.db` 或租户目录）
  - `HUMAN_SUPPORT_URL`：人工兜底渠道 URL（可选）
  - `DASHSCOPE_API_KEY`：阿里云通义 API Key（如启用 Qwen）
  - `REDIS_URL`：`redis://host:port/db`（可选，无则回退内存）
  - `TENANTS_BASE_DIR`：多租户根目录（默认 `work/tenants`）
  - `COURSE_TENANT_MAP`：课程到租户映射文件（默认 `work/tenant_courses.json`）

## 初始化步骤
- 订单库初始化（可选，如需样例数据）：
  - 本地：`python -m work.init_orders_db`
  - Docker：进入容器后执行同命令或在宿主机执行 `python work/init_orders_db.py`
- 知识库索引构建（可选）：
  - 本地：`python -m work.rag-train --tenant default --datas work/datas`
  - 租户：`python -m work.rag-train --tenant t1 --datas work/tenants/t1/datas`
- 日志推送守护（可选）：
  - `python work/log_push.py --daemon --config work/log_push_config.json`
  - 验证：查看 `work/logs/log_push_service.log`

## 启动服务
- 本地启动（推荐命令）：`python -m uvicorn work.app:app --host 0.0.0.0 --port 8000 --reload`
- 备用命令：在 `work` 目录下执行 `uvicorn app:app --host 0.0.0.0 --port 8000`
- 生产模式：`python -m uvicorn work.app:app --host 0.0.0.0 --port 8000 --workers 2`

## 验证方法
- 健康检查：`curl http://localhost:8000/health`
- 对话接口：
  ```bash
  curl -X POST http://localhost:8000/chat \
       -H "Content-Type: application/json" \
       -d '{"query":"课程适合新手吗？","thread_id":"demo-thread"}'
  ```
- 建议问题 SSE：`curl -N http://localhost:8000/suggest/demo-thread`
- 订单查询（支持租户）：`curl http://localhost:8000/api/orders/20251114001 -H "X-Tenant-ID: default"`
- 向量添加（鉴权）：
  ```bash
  curl -X POST http://localhost:8000/api/v1/vectors/items \
       -H "Content-Type: application/json" -H "X-API-Key: test" \
       -d '{"items":[{"text":"示例文本","metadata":{"source":"api"}}]}'
  ```

## 环境差异配置
- 开发环境：
  - `--reload` 自动重载；日志级别 `INFO`；可省略 Redis；使用默认租户 `default`。
  - KB 与 DB 使用本地路径：`work/faiss_index`、`work/db/orders.sqlite`。
- 测试环境：
  - 固定依赖版本；启用 Redis；准备独立测试租户与数据集。
  - 使用 `docker-compose` 统一编排与健康检查；通过 `.env.test` 管理配置。
- 生产环境：
  - 多副本/多进程（`--workers`）；前置反向代理与 TLS；持久化日志至外部系统（ELK）。
  - 明确 `TENANTS_BASE_DIR` 与备份策略；定期构建与校验 FAISS 索引。

## 故障排查
- 应用不可用：检查 `docker-compose logs app` 或本地控制台；验证 `/health`。
- 向量检索失败：确认 `faiss-cpu` 安装与索引路径；查看 `work/faiss_index/index.faiss`。
- 订单查询 500：检查 `ORDERS_DB_PATH` 配置与文件存在；用 `sqlite3` 验证 SQL。
- SSE 超时：前端需保持连接，服务端默认 15s 超时；查看 `SUGGEST_QUEUES` 是否有事件。
- 日志推送失败：查看 `work/logs/log_push_service.log` 与 `log_push_config.json` 鉴权参数。

## 日志与监控
- 应用日志：`work/logs/requests.log`（包含耗时、模型切换、向量增删审计）
- 指标快照：`GET /health` 返回 `overall/kb/order/direct/handoff/vectors_*` 的最值与分位统计
- 外部推送：`log_push.py` 守护将日志以 NDJSON 上报 ELK/Logstash

## 关联文档
- 架构设计：参见《[architecture_design.md](./architecture_design.md)》
- API 规范：参见《[api_specification.md](./api_specification.md)》
- 性能报告：参见《[performance_report.md](./performance_report.md)》