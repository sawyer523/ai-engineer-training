1. 健康检查
curl.exe -s http://127.0.0.1:8000/health

2. 课程/售前/售后（检索知识库）
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"新手能学吗？\"}"

3. 订单查询（数据库工具）
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"查询订单 20251114001 什么时候开课？\"}"

4. 人工转接
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"转人工\"}"

5. 订单查询 HTTP API
curl.exe -s http://127.0.0.1:8000/api/orders/20251114001
====
6. 开场白（问候与预置选项）
curl.exe -s http://127.0.0.1:8000/greet

7. 快捷指令：/help
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"/help\",\"thread_id\":\"abc123\"}"

8. 查看上下文历史：/history（需先产生对话）
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"小白适合学这门课吗？\",\"thread_id\":\"abc123\"}"
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"那Java程序员呢？\",\"thread_id\":\"abc123\"}"
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"/history\",\"thread_id\":\"abc123\"}"

9. 重置上下文：/reset
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"/reset\",\"thread_id\":\"abc123\"}"
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"/history\",\"thread_id\":\"abc123\"}"

10. 推送建议
# 先触发对话，随后订阅建议（同一 thread_id）
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"课程用到了哪些大模型工具\",\"thread_id\":\"abc123\"}"
curl.exe --no-buffer http://127.0.0.1:8000/suggest/abc123
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"课程用到了哪些开发框架\",\"thread_id\":\"abc123\"}"

11. 模型查看
curl.exe -s http://127.0.0.1:8000/models/list 

12. 模型切换
curl.exe -s -X POST http://127.0.0.1:8000/models/switch -H "Content-Type: application/json; charset=utf-8" -d "{\"name\":\"qwen-turbo\"}"

13. mcp接口的支持
curl.exe -i -N http://127.0.0.1:8000/mcp/sse

data: /mcp/messages/?session_id=e4d5f80531e54c0e8b6be7c3ea9a42ea

curl -s -X POST "http://127.0.0.1:8000/mcp/messages/?session_id=e4d5f80531e54c0e8b6be7c3ea9a42ea" -H "Content-Type: application/json; charset=utf-8" --data-binary "{ \"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"kb_search\",\"arguments\":{\"query\":\"课程适合新手吗？\"}} }"


14. 多模态问题的支持
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"查询订单 20251114001 什么时候开课？\",\"images\":[\"https://example.com/order.png\"],\"thread_id\":\"abc123\"}"

curl.exe -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json; charset=utf-8" ^
  -d "{\"audio\":\"https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3\"}"

15. 增加新的条目到知识库(指定id)
curl -X POST "http://127.0.0.1:8000/api/v1/vectors/items" -H "Content-Type: application/json" -H "X-API-Key: test" --data-raw "{\"items\":[{\"text\":\"101. **Q：课程会讲解哪种向量数据 库？**  \\nA：课程已包含 FAISS 的向量检索，支持自然语言查询并返回最相关的问答片段及来源路径。\",\"metadata\":{\"source\":\"tests\"},\"id\":\"fd6580e206b7b768dc305c316c5ae894cfa6a877\"}]}" 

16. 增加新的条目到知识库(不指定id，由服务端生成 sha1(text)）)
curl -X POST "http://127.0.0.1:8000/api/v1/vectors/items" -H "Content-Type: application/json" -H "X-API-Key: YOUR_API_KEY" --data-raw "{\"items\":[{\"text\":\"101. **Q：课程会讲解哪种向量数据库？**  \\nA：课程已包含 FAISS 的向量检索，支持自然语言查询并返回最相关的问答片段及来源路径。\",\"metadata\":{\"source\":\"tests\"}}]}"

17. 批量新增（混合指定/不指定 id）
curl -X POST "http://127.0.0.1:8000/api/v1/vectors/items" -H "Content-Type: application/json" -H "X-API-Key: YOUR_API_KEY" --data-raw "{\"items\":[{\"text\":\"101. **Q：课程会讲解哪种向量数据库？**  \\nA：课程已包含 FAISS 的向量检索，支持自然语言查询并返回最相关的问答片段及来源路径。\",\"metadata\":{\"source\":\"tests\"},\"id\":\"fd6580e206b7b768dc305c316c5ae894cfa6a877\"},{\"text\":\"FAISS 支持自然语言查询示例\",\"metadata\":{\"source\":\"tests\"}}]}"

18. 删除（按 id）
curl -X DELETE "http://127.0.0.1:8000/api/v1/vectors/items" -H "Content-Type: application/json" -H "X-API-Key: test" --data-raw "{\"ids\":[\"fd6580e206b7b768dc305c316c5ae894cfa6a877\"]}"
=====
19. 多租户测试
- 知识库添加
curl.exe -s -X POST http://127.0.0.1:8000/api/v1/vectors/items -H "Content-Type: application/json; charset=utf-8" -H "X-API-Key: test" -H "X-Tenant-ID: t1" --data-raw "{\"items\":[{\"text\":\"101. Q：t1租户示例问答\\nA：这是t1租户的知识库新增示例。\",\"metadata\":{\"source\":\"README\"},\"id\":\"t1_kb_demo_002\"}]}"

- 知识库查询
curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json; charset=utf-8" -H "X-Tenant-ID: t1" -d "{\"query\":\"是t1租户的课程吗？\"}"


新增能力清单
- 用户交互
  - 开场白：系统在启动时，会向用户打招呼，询问用户需要咨询的问题。
  - 开场白预置问题
    - 课程咨询
    - 订单查询
    - 人工转接
  - 用户问题建议
    - ReAct显式“思考→行动→观察”流程
  - 快捷指令

- 工具
  - 本地模型切换
  - mcp接口的支持

- 知识库管理
  - 缺少对word、pdf、excel等文件格式的支持
  - 备份能力
  - 知识库的增量、删除

- 多模态问题的支持
  - 缺少对多模态问题（如图片、语音等）的支持

- 部署
  - 健康检查和指标监控
  - 对docker的支持
  - 远程日志记录

- 多租户支持
  - 不同客户/品牌可以在同一平台上部署，互不干扰。
  - 每个客户/品牌可以有自己的知识库、模型配置和用户数据。

- 平台化
  - 完整的 Agent 平台源码包（含 Web 前后端 + 移动 App ）

- 文档
  - 架构设计
  - 部署说明
  - API 文档
  - 性能报告
  - 演示视频：
    - 展示多 Agent 协作
    - 任务编排
    - 移动端推理等核心功能

- 第三方接入
  - 支持钉钉、微信公众号等接口接入


