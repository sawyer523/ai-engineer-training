# 日志推送到 ELK 使用说明

## 概述
- 将 `work/logs/requests.log` 的新增内容增量推送到 Logstash HTTP 接口，供 Elasticsearch/Kibana 分析。
- 采用模块化设计，包含日志读取、解析、HTTP 推送、状态管理与健康检查。
- 支持日志轮转检测、批量推送、多线程处理与可配置字段映射。

## 前置条件
- 已安装并运行 Logstash，启用 HTTP 输入插件。例如：

```
input {
  http {
    host => "0.0.0.0"
    port => 8080
    codec => json_lines
  }
}
filter {
}
output {
  elasticsearch {
    hosts => ["http://localhost:9200"]
    index => "requests-log-%{+YYYY.MM.dd}"
  }
}
```

## 配置
- 文件：`work/log_push_config.json`
- 关键字段：
  - `input_file`：输入日志文件路径。
  - `state_file`：增量状态持久化文件路径。
  - `logstash_http_url`：Logstash HTTP 接口地址。
  - `auth.type`：`none`、`basic`、`bearer`。
  - `batch_size`、`batch_flush_ms`：批量大小与刷新时间。
  - `field_map`：字段映射，例如将 `timestamp` 映射到 `@timestamp`。
  - `metadata`：附加元数据，如 `source`。

示例：见已生成的 `work/log_push_config.json`。

## 运行
- 手动一次性增量推送：
  - `python work/log_push.py --mode manual --config work/log_push_config.json`
  - 仅预览不推送：`python work/log_push.py --mode manual --dry-run`
  - 从文件开头重新推送：`python work/log_push.py --mode manual --from-start`

- 守护模式持续跟踪推送：
  - `python work/log_push.py --mode daemon --config work/log_push_config.json`
  - 支持轮转检测与增量处理，按批量与时间窗口刷新。

- 健康检查：
  - `python work/log_push.py --mode health`
  - 发送一条测试事件到 Logstash，检查连通性与 HTTP 接口可用性。

## 字段解析与映射
- 默认解析日志行格式：`YYYY-MM-DD HH:MM:SS,mmm | LEVEL | LOGGER | MESSAGE`。
- 对 `MESSAGE` 中的 `key=value` 进行提取，如 `route=course`、`cost=2756.59ms`。
- 自动解析 `request={...}` JSON，提取 `query` 到 `request.query`（可通过 `field_map` 调整）。
- 添加 `@timestamp`、`host`、`source` 等元数据。可在配置中扩展。

## 兼容性
- 保留原 `setup_logger`、`generate_test_logs`、`test_elk_connection` 方法以便原有测试流程使用。
- 推送方式由 TCP 调整为 HTTP；如需继续使用旧方式，可独立保留旧脚本。

## 日志与状态
- 运行日志：`work/logs/log_push_service.log`。
- 增量状态：`work/state/log_push_state.json`。可删除以重置增量位置。

## 常见问题
- 无法连接 Logstash：检查 `logstash_http_url`、端口与网络连通性，确认 HTTP 输入插件已启用且使用 `json_lines`（NDJSON）。
- 字段未正确解析：调整 `field_map` 或更改日志格式解析规则。
- 重复数据：删除或编辑 `state_file` 重置位置；或使用 `--from-start` 重新推送。