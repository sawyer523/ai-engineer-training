# API 规范说明

## 总览
- 基础 URL：`http://<host>:8000`
- 认证：
  - 向量库管理需 `X-API-Key: test`
  - 多租户通过 `X-Tenant-ID: <tenant>` 或 `?tenant=<tenant>`
- 内容类型：`application/json`
- 状态码约定：
  - 200 成功；401 未授权；404 资源不存在；400 参数错误；500 服务内部错误
- 统一响应包（部分接口）：`{code, message, data}`；业务接口返回领域对象

## 1. 对话接口
- `POST /chat`
- 功能：基于 LangGraph 路由与工具层，返回答案与来源；支持图片/音频触发多模模型
- 请求体：
  ```json
  {
    "query": "课程适合新手吗？",
    "user_id": "u-001",
    "thread_id": "t-abc",
    "images": ["http://.../img1.png"],
    "audio": "base64-wav-or-url",
    "asr_language": "zh",
    "asr_itn": true
  }
  ```
- 响应体：
  ```json
  {
    "route": "course", // 或 order/direct/human
    "answer": "课程为新手准备了...",
    "sources": [
      {"source": "datas/data.txt", "content": "..."}
    ]
  }
  ```
- 备注：若输入以 `/help` `/history` `/reset` 开头，则返回命令结果而非标准对话结构
- 示例：
  ```bash
  curl -X POST http://localhost:8000/chat \
       -H "Content-Type: application/json" \
       -H "X-Tenant-ID: default" \
       -d '{"query":"课程适合新手吗？","thread_id":"demo"}'
  ```

## 2. 建议问题推送（SSE）
- `GET /suggest/{thread_id}`
- 功能：异步推送建议问题事件流；事件类型 `react_start`/`react`/`error`
- 响应：`text/event-stream`
- 事件示例：
  ```text
  id: demo
  event: react_start
  data: {"route":"course","suggestions":[],"event":"react_start"}

  id: demo
  event: react
  data: {"route":"course","suggestions":["课程目录","学习周期","适合人群"],"final":true}
  ```
- 示例：`curl -N http://localhost:8000/suggest/demo`

## 3. 模型管理
- `GET /models/list`：返回当前模型与可选模型
  - 响应：`{"code":0,"message":"OK","data":{"current":"qwen-turbo","models":["qwen-turbo","qwen-plus"]}}`
- `POST /models/switch`
  - 请求体：`{"name":"qwen-plus"}`
  - 响应：`{"code":0,"message":"OK","data":{"current":"qwen-plus","models":[...]}}`
  - 说明：切换后刷新 LangGraph 链；失败返回 `{"code":"error","message":"..."}`

## 4. 向量库管理（鉴权）
- 公共要求：`X-API-Key: test`
- `POST /api/v1/vectors/items`（添加文本向量）
  - 请求体：
    ```json
    {
      "items": [
        {"text": "示例文本", "metadata": {"source": "api"}, "id": "opt-id"}
      ]
    }
    ```
  - 响应：`{"code":0,"message":"OK","data":{"added":1,"ids":["..."],"skipped":["..."]}}`
  - 示例：
    ```bash
    curl -X POST http://localhost:8000/api/v1/vectors/items \
         -H "Content-Type: application/json" -H "X-API-Key: test" \
         -d '{"items":[{"text":"示例","metadata":{"source":"api"}}]}'
    ```
- `DELETE /api/v1/vectors/items`（删除文本向量）
  - 请求体：`{"ids":["id1","id2"]}`
  - 响应：`{"code":0,"message":"OK","data":{"deleted":2,"ids":["id1","id2"]}}`

## 5. 订单查询
- `GET /api/orders/{order_id}`
- 功能：查询订单状态与关键字段（租户优先）
- 响应示例：
  ```json
  {
    "order_id": "20251114001",
    "status": "PAID",
    "amount": 1999.0,
    "updated_at": "2025-11-14 12:00:00",
    "enroll_time": null,
    "start_time": "2025-11-20 19:00:00"
  }
  ```
- 错误：
  - 404 `{"detail":"Order not found"}`
  - 500 `{"detail":"Orders database not configured"}` 或 `{"detail":"Internal Server Error"}`
- 示例：`curl http://localhost:8000/api/orders/20251114001 -H "X-Tenant-ID: default"`

## 6. 欢迎接口
- `GET /greet`
- 响应：包含常用入口 `course/order/human`

## 7. 健康检查
- `GET /health`
- 响应：
  ```json
  {
    "model": "qwen-turbo",
    "kb_index": true,
    "orders_db": true,
    "metrics": {
      "overall": {"count": 120, "min": 5.0, "max": 4800.0, "avg": 2600.0, "p95": 4200.0},
      "kb": {"...": "..."},
      "order": {"...": "..."},
      "direct": {"...": "..."},
      "handoff": {"...": "..."},
      "vectors_add": {"...": "..."},
      "vectors_delete": {"...": "..."}
    }
  }
  ```
- 说明：指标快照由服务运行期累积；具体数值依环境与负载而变。

## 认证与授权
- API Key：仅 `POST/DELETE /api/v1/vectors/items` 需要 `X-API-Key: test`
- 多租户：优先使用 `X-Tenant-ID`；未提供时可用 `?tenant=`；默认 `default`
- 请求 ID：服务为每个请求注入 `X-Request-Id`，用于链路追踪；可在响应头获取

## 错误处理
- 统一错误：
  - API 包装：`{"code":"invalid","message":"无效模型"}`、`{"code":"graph_reload_error","message":"..."}`
  - HTTP 异常：`{"detail":"Unauthorized"}`、`{"detail":"Order not found"}`、`{"detail":"Internal Server Error"}`
- 脱敏：中间件会对 JSON 体中的敏感字段与模式进行脱敏（例如身份证、密码在日志中显示为 `[REDACTED]`）

## 代码示例
- Python（对话）：
  ```python
  import requests
  r = requests.post("http://localhost:8000/chat", json={"query":"课程适合新手吗？","thread_id":"demo"})
  print(r.json())
  ```
- Python（向量添加）：
  ```python
  import requests
  r = requests.post(
      "http://localhost:8000/api/v1/vectors/items",
      headers={"X-API-Key":"test"},
      json={"items":[{"text":"示例文本","metadata":{"source":"api"}}]}
  )
  print(r.json())
  ```

## 关联文档
- 架构设计：参见《[architecture_design.md](./architecture_design.md)》
- 部署说明：参见《[deployment_guide.md](./deployment_guide.md)》
- 性能报告：参见《[performance_report.md](./performance_report.md)》