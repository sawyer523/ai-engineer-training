# ELK 日志系统整合使用说明

## 概述

本项目实现了与 ELK (Elasticsearch, Logstash, Kibana) 日志系统的完整整合，支持实时日志传输、处理和可视化。

## 系统架构

```
Python 应用 (p41elk.py) 
    ↓ TCP 连接 (端口 5044)
Logstash (日志处理)
    ↓ 存储
Elasticsearch (日志存储)
    ↓ 查询和可视化
Kibana (Web 界面)
```

## 快速开始

### 1. 启动 ELK 服务

```bash
cd elk
docker-compose up -d
```

### 2. 验证服务状态

- Elasticsearch: http://localhost:9200
- Kibana: http://localhost:5601
- Logstash: 监听端口 5044 和 5959

### 3. 运行日志生成程序

```bash
python p41elk.py
```

## 功能特性

### 1. 多重日志输出
- **控制台输出**: 实时查看日志信息
- **文件输出**: 保存到 `elk_integration.log`
- **Logstash 输出**: 发送到 ELK 系统

### 2. 自定义 Logstash TCP 处理器
- **自动重连**: 连接断开时自动重试
- **错误处理**: 完善的异常处理机制
- **线程安全**: 支持多线程环境
- **JSON 格式**: 结构化日志数据

### 3. 日志生成功能
- **持续生成**: 默认 10 分钟，可自定义
- **定时输出**: 每秒生成一条日志
- **多种级别**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **丰富内容**: 包含时间戳、模块、函数、行号等信息

## 配置说明

### Logstash 配置 (elk/config/logstash.conf)

```ruby
input {  
  tcp {
    port => 5959
    codec => json
  }
  tcp {
    port => 5044
    codec => json_lines
  }
}
output {
  elasticsearch { hosts => ["elasticsearch:9200"] }
}
```

### Python 日志配置

```python
# 自定义 TCP 处理器参数
LogstashTCPHandler(
    host='localhost',      # Logstash 地址
    port=5044,            # 连接端口
    timeout=5,            # 连接超时
    max_retries=3         # 最大重试次数
)
```

## 日志格式

发送到 Logstash 的日志采用 JSON 格式：

```json
{
    "timestamp": "2024-01-15T10:30:45.123456",
    "level": "INFO",
    "logger": "elk_integration",
    "message": "用户登录成功 - 序号: 1, 时间: 10:30:45",
    "module": "p41elk",
    "function": "generate_test_logs",
    "line": 245,
    "thread": 12345,
    "process": 6789
}
```

## Kibana 使用

### 1. 创建索引模式

1. 访问 http://localhost:5601
2. 进入 "Stack Management" > "Index Patterns"
3. 创建索引模式 `logstash-*`
4. 选择时间字段 `@timestamp`

### 2. 查看日志

1. 进入 "Discover" 页面
2. 选择创建的索引模式
3. 设置时间范围查看日志

### 3. 创建可视化

1. 进入 "Visualize" 页面
2. 创建各种图表和仪表板
3. 监控日志级别分布、时间趋势等

## 故障排除

### 1. 连接问题

如果出现连接 Logstash 失败：

```bash
# 检查 Docker 容器状态
docker-compose ps

# 查看 Logstash 日志
docker-compose logs logstash

# 检查端口占用
netstat -an | findstr 5044
```

### 2. 日志不显示

1. 确认 ELK 服务正常运行
2. 检查 Kibana 索引模式配置
3. 验证时间范围设置
4. 查看 Elasticsearch 索引状态

### 3. 性能优化

- 调整日志级别减少输出量
- 增加 Logstash 内存配置
- 优化 Elasticsearch 索引设置

## 扩展功能

### 1. 添加更多日志源

```python
# 添加其他应用的日志处理器
app_logger = logging.getLogger('my_app')
app_logger.addHandler(logstash_handler)
```

### 2. 自定义日志字段

```python
# 在 LogstashTCPHandler.emit() 中添加自定义字段
log_entry.update({
    'environment': 'production',
    'service': 'web-api',
    'version': '1.0.0'
})
```

### 3. 日志过滤和转换

在 Logstash 配置中添加 filter 部分：

```ruby
filter {
  if [level] == "ERROR" {
    mutate {
      add_tag => ["error"]
    }
  }
}
```

## 注意事项

1. **网络连接**: 确保 Python 程序能访问 Logstash 端口
2. **资源使用**: 长时间运行会产生大量日志数据
3. **时区设置**: 注意日志时间戳的时区配置
4. **安全考虑**: 生产环境建议启用 ELK 安全功能

## 技术支持

如遇问题，请检查：
- Docker 容器运行状态
- 网络连接和端口配置
- 日志文件和错误信息
- ELK 各组件的健康状态