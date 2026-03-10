from typing import TypedDict, List, Any


class AgentState(TypedDict):
    """定义图的状态"""
    topic: str
    style: str
    length: int
    research_report: str
    draft: str
    review_suggestions: str
    final_article: str
    log: List[str]
    error_count: int
    max_retries: int
