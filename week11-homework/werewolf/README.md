# 狼人杀游戏系统 - 使用说明

## 快速开始

### 1. 安装依赖
```bash
uv sync
```

### 2. 配置 API 密钥
```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 DeepSeek API Key：
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### 3. 运行游戏
```bash
python -m werewolf.main
```

或
```bash
cd werewolf && python main.py
```

## 系统特性

### ✅ 已实现功能

| 功能 | 状态 | 说明 |
|------|------|------|
| Agent 角色建模 | ✅ | 狼人/村民 Prompt 模板，6种性格特质 |
| 游戏流程控制 | ✅ | Moderator Agent 协调阶段流转 |
| 记忆管理 | ✅ | 情景记忆+语义记忆，FAISS 向量检索 |
| RAG 增强推理 | ✅ | 语义相似度检索历史记忆辅助决策 |
| 可视化执行追踪 | ✅ | Thought/Action/Observation 日志 |
| 成本分析 | ✅ | Token 统计、预估成本 |

### 玩家配置

当前配置（5人局）：
- 3 名村民：观察型、谨慎型、激进型
- 2 名狼人：魅力型、多疑型

可在 `werewolf/main.py` 中修改配置。

## 游戏流程

```
游戏开始 → 夜晚讨论 → 夜晚行动 → 天亮公布 → 发言环节 → 投票处决 → 循环
```

## 输出文件

游戏结束后生成：
- `game_logs/game_YYYYMMDD_HHMMSS.json`: 完整游戏日志
- 终端输出：实时游戏过程和成本分析

## 项目结构

```
werewolf/
├── agents/          # Agent 实现
│   ├── player.py    # 玩家 Agent
│   └── moderator.py # 主持人 Agent
├── game/            # 游戏状态管理
│   └── state.py     # 游戏状态定义
├── memory/          # 记忆管理
│   └── manager.py   # 记忆存储和检索
├── prompts/         # Prompt 模板
│   └── templates.py # 角色和阶段 Prompt
├── utils/           # 工具函数
│   └── logger.py    # 日志系统
├── main.py          # 入口文件
└── DESIGN.md        # 设计文档
```

## 技术栈

- **框架**: LangChain + LangGraph
- **模型**: DeepSeek-Chat
- **向量数据库**: FAISS
- **日志**: Rich + JSON

## 设计文档

详见 [DESIGN.md](DESIGN.md)

## 成本说明

DeepSeek-Chat 定价（仅供参考）：
- 输入: ¥0.14 / 1M tokens
- 输出: ¥0.28 / 1M tokens

单局游戏（5人，约3-5轮）预估成本：¥0.01-0.05

## 常见问题

**Q: 如何添加更多玩家？**
A: 在 `main.py` 的 `player_configs` 中添加更多配置。

**Q: 如何修改性格特质？**
A: 在 `prompts/templates.py` 的 `PERSONALITY_TRAITS` 中添加或修改。

**Q: 游戏日志在哪里？**
A: 保存在 `game_logs/` 目录下。

**Q: 如何查看某个玩家的记忆？**
A: 可以在代码中调用 `agent.memory.get_context_string()` 查看。

## License

MIT
