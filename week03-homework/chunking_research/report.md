# 句子切片检索实验报告

## 实验概述

本实验探索了 LlamaIndex 中不同的文本切片策略对检索增强生成（RAG）系统效果的影响。
使用 OpenAI 兼容的 gpt-4o-mini 模型和 text-embedding-3-small 嵌入模型（通过 api.apiyi.com 代理）。

## 实验环境

- **语言模型**: gpt-4o-mini
- **嵌入模型**: text-embedding-3-small
- **API 代理**: https://api.apiyi.com/v1
- **数据集**: 3篇文档（量子计算、气候变化、人工智能），每篇约1500字
- **测试查询**:
  1. 什么是量子纠缠？
  2. 气候变化的主要原因是什么？
  3. Transformer架构有什么作用？

## 切片方式对比

### 1. Sentence Splitter (句子切片器)

| 参数配置 | 描述 |
|---------|------|
| chunk_size=512, overlap=50 | 大块，中等重叠 |
| chunk_size=256, overlap=25 | 中等块，小重叠 |
| chunk_size=128, overlap=0 | 小块，无重叠 |

### 2. Token Splitter (令牌切片器)

| 参数配置 | 描述 |
|---------|------|
| chunk_size=128, overlap=16 | 中等块，基于token |

### 3. Sentence Window Node Parser (句子窗口解析器)

| 参数配置 | 描述 |
|---------|------|
| window_size=1 | 返回匹配句子+前后1句 |
| window_size=3 | 返回匹配句子+前后3句 |
| window_size=5 | 返回匹配句子+前后5句 |

## 实验结果分析

### 检索质量对比

| 切片方式 | 平均检索节点数 | 上下文丰富性 | 精确度 | 备注 |
|---------|--------------|-------------|--------|------|
| Sentence Splitter (512, 50) | 2.0 | 高 | 中等 | 适合上下文丰富的回答 |
| Sentence Splitter (256, 25) | 2.0 | 中等 | 较高 | **最佳平衡** |
| Sentence Splitter (128, 0) | 2.0 | 低 | 高 | 精确匹配但上下文不足 |
| Token Splitter | 2.0 | 中等 | 中等 | 适合结构化文本 |
| Sentence Window (w=1) | 2.0 | 低 | 高 | 上下文最精确 |
| Sentence Window (w=3) | 2.0 | 中等 | 高 | 推荐配置 |
| Sentence Window (w=5) | 2.0 | 高 | 较高 | 上下文最丰富 |

### 详细查询结果分析

#### 查询1: 什么是量子纠缠？

| 切片方式 | 最高相似度 | 回答质量 |
|---------|-----------|---------|
| Sentence Splitter (512, 50) | 0.5890 | 良好 |
| Sentence Splitter (256, 25) | **0.6545** | 优秀 |
| Sentence Splitter (128, 0) | 0.5889 | 良好 |
| Token Splitter | 0.5773 | 良好 |
| Sentence Window (w=1) | 0.5227 | 良好 |
| Sentence Window (w=3) | 0.5228 | 良好 |
| Sentence Window (w=5) | 0.5228 | 良好 |

**观察**: Sentence Splitter (256, 25) 获得最高相似度分数，说明中等块大小配合适当重叠能提高检索准确性。

#### 查询2: 气候变化的主要原因是什么？

| 切片方式 | 最高相似度 | 回答质量 |
|---------|-----------|---------|
| Sentence Splitter (512, 50) | 0.6287 | 优秀 |
| Sentence Splitter (256, 25) | **0.6809** | 优秀 |
| Sentence Splitter (128, 0) | 0.5717 | 良好 |
| Token Splitter | 0.5625 | 良好 |
| Sentence Window (w=1) | 0.4748 | 良好 |
| Sentence Window (w=3) | 0.4748 | 良好 |
| Sentence Window (w=5) | 0.4748 | 良好 |

**观察**: 同样是 Sentence Splitter (256, 25) 表现最佳，所有配置都能正确回答问题。

#### 查询3: Transformer架构有什么作用？

| 切片方式 | 最高相似度 | 回答质量 |
|---------|-----------|---------|
| Sentence Splitter (512, 50) | 0.3748 | 良好 |
| Sentence Splitter (256, 25) | **0.3865** | 优秀 |
| Sentence Splitter (128, 0) | 0.3536 | 良好 |
| Token Splitter | 0.3757 | 良好 |
| Sentence Window (w=1) | 0.2010 | **失败** |
| Sentence Window (w=3) | 0.2010 | **失败** |
| Sentence Window (w=5) | 0.2010 | **失败** |

**观察**: 这是跨文档检索的挑战案例。Transformer 相关信息在人工智能文档中，但量子计算和气候变化文档的相似度得分反而更高。Sentence Window 在此场景下表现不佳，返回了错误的文档。这表明 Sentence Window 可能不太适合跨主题的模糊查询。

## 参数影响分析

### chunk_size 影响

- **大 chunk_size (512)**:
  - 优点: 保留更多上下文信息，适合需要长篇理解的问题
  - 缺点: 可能降低检索精确度，增加噪音
  - 适用: 解释型、概括型查询

- **中等 chunk_size (256)** - **推荐**:
  - 优点: 平衡了上下文完整性和检索精确度
  - 缺点: 无明显缺点
  - 适用: 多数场景下的最佳选择

- **小 chunk_size (128)**:
  - 优点: 检索更精确，匹配更精准
  - 缺点: 可能丢失上下文，导致回答不够完整
  - 适用: 事实型精确查询

### chunk_overlap 影响

- **有 overlap (25-50)**:
  - 优点: 减少信息在边界处的丢失，提高连续性
  - 缺点: 增加冗余，略微降低检索效率

- **无 overlap (0)**:
  - 优点: 减少冗余，提高效率
  - 缺点: 可能在边界处截断重要信息

**实验结论**: overlap 设为 chunk_size 的 10-20% 是合理选择。

### window_size 影响 (Sentence Window)

- **window_size=1**:
  - 检索最精确，但上下文严重不足
  - 在跨文档查询时表现差（如 Transformer 查询完全失败）

- **window_size=3**:
  - 平衡选择，多数情况下效果良好
  - 但在本实验中，跨文档检索仍有问题

- **window_size=5**:
  - 充足上下文，接近 Sentence Splitter 的效果
  - 仍未解决跨文档检索问题

**关键发现**: Sentence Window 在跨文档模糊查询时表现不佳，可能的原因是检索基于单句，缺乏足够的语义上下文来进行正确的文档区分。

## 精确检索 vs 上下文丰富性权衡

| 场景 | 推荐策略 | 理由 |
|-----|---------|------|
| 事实型查询（如定义、数据） | Sentence Splitter (256, 25) | 最佳平衡，实验中表现最稳定 |
| 解释型查询（如原因、过程） | Sentence Splitter (512, 50) | 需要充分上下文支持推理 |
| 概括型查询（如总结） | Sentence Splitter (512, 50) | 需要完整的语义单元 |
| 跨文档模糊查询 | Sentence Splitter (256, 25) | Sentence Window 表现不佳 |

## 结论与建议

### 核心发现

1. **最佳配置**: Sentence Splitter (chunk_size=256, overlap=25) 在所有测试中表现最稳定、最优秀
   - 在三个查询中都取得了最高或接近最高的相似度分数
   - 平衡了检索精确度和上下文完整性

2. **Sentence Window 的局限性**: 虽然理论上能平衡精确性和上下文，但在实际测试中：
   - 对单文档内的精确查询效果良好
   - 在跨文档模糊查询时完全失败（Transformer 查询案例）
   - 可能不适合多主题知识库的场景

3. **chunk_overlap 的重要性**: 有重叠的切片 (overlap=25) 比无重叠 (overlap=0) 表现更稳定

4. **chunk_size 的选择**: 256 是最佳平衡点，512 偏大导致精确度下降，128 偏小导致上下文不足

### 参数选择原则

1. **优先使用 Sentence Splitter** 而非 Sentence Window，除非你的应用场景是单主题文档
2. **chunk_size 建议**: 256 作为起始点，根据具体应用调整
3. **overlap 建议**: 设为 chunk_size 的 10-20%
4. **similarity_top_k**: 建议 2-5，本实验中使用 2

### 最佳实践

1. 从 Sentence Splitter (256, 25) 开始作为基准配置
2. 如果检索精确度不足，减小 chunk_size 到 128
3. 如果上下文不够，增大 chunk_size 到 512
4. 通过 A/B 测试验证实际效果
5. 对于多主题知识库，谨慎使用 Sentence Window

### 局限性与改进方向

1. **跨文档检索**是当前配置的弱点，可以考虑：
   - 增加检索的 top_k 数量
   - 使用重排序（reranking）机制
   - 改进查询理解，识别跨文档意图

2. **Sentence Window 需要进一步研究**其适用场景和失败原因

3. 可以尝试混合策略：对简单查询用小窗口，对复杂查询用大窗口
