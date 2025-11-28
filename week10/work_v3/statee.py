"""状态定义模块：用于在 LangGraph 流程中传递会话状态。

包含用户查询、意图路由、知识库回答、RAG 来源、订单摘要、
人工交接信息与最终路由等字段。
"""
from typing import List, Dict, Any
from typing_extensions import TypedDict, Literal


class State(TypedDict, total=False):
    """会话状态字典类型

    - query：用户原始问题
    - history：最近对话摘要
    - intent：意图路由标签（course/presale/postsale/order/human/direct）
    - kb_answer：知识库检索或直答的结果文本
    - sources：RAG 检索到的文档来源（metadata 列表）
    - order_summary：订单查询生成的客服话术摘要
    - human_handoff：转人工时的渠道与载荷
    - route：最终路由落点（冗余字段，便于返回）
    - tenant_id：当前租户标识
    """
    query: str
    history: str
    intent: Literal["course", "presale", "postsale", "order", "human", "direct"]
    kb_answer: str
    sources: List[Dict[str, Any]]
    order_summary: str
    human_handoff: Dict[str, Any]
    route: str
    tenant_id: str
