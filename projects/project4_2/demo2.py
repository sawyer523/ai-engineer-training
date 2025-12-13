from typing import List, Dict, Any, Optional, TypedDict, Annotated, Callable
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    order_id: Optional[str]

class MockLLM:
    def invoke(self, messages):
        return AIMessage(content="我可以帮您查询订单。")

# --- 动态图管理器 (Dynamic Graph Manager) ---

class DynamicGraphManager:
    """
    负责管理 LangGraph 的动态构建、节点注册与热重载。
    """
    def __init__(self):
        self.nodes: Dict[str, Callable] = {}
        self.edges: List[tuple] = []
        self.entry_point: str = ""
        self.app = None
        self.checkpointer = MemorySaver()
        self._dirty = False
        
        # 初始化基础节点
        self._init_base_nodes()

    def _init_base_nodes(self):
        llm = MockLLM()
        def chatbot_node(state: AgentState):
            return {"messages": [llm.invoke(state["messages"]) ]}
        def input_processing_node(state: AgentState):
            return {"messages": []}
        self.register_node("input_proc", input_processing_node)
        self.register_node("chatbot", chatbot_node)
        self.entry_point = "input_proc"
        self.add_edge("input_proc", "chatbot")
        self._dirty = True

    def register_node(self, name: str, func: Callable):
        """注册新节点"""
        self.nodes[name] = func
        print(f"[GraphManager] 节点已注册: {name}")
        self._dirty = True

    def add_edge(self, start: str, end: str):
        """添加普通边"""
        self.edges.append((start, end))
        self._dirty = True

    

    def clear_edges_from(self, source_node: str):
        """辅助方法：清除从指定节点出发的所有边"""
        self.edges = [e for e in self.edges if e[0] != source_node]
        print(f"[GraphManager] 已清除节点 {source_node} 的出边")
        self._dirty = True

    def _compile_if_needed(self):
        if not self._dirty and self.app is not None:
            return
        print("[GraphManager] 正在重新编译图...")
        workflow = StateGraph(AgentState)
        for name, func in self.nodes.items():
            workflow.add_node(name, func)
        for start, end in self.edges:
            workflow.add_edge(start, end)
        if self.entry_point:
            workflow.add_edge(START, self.entry_point)
        self.app = workflow.compile(checkpointer=self.checkpointer)
        self._dirty = False
        print("[GraphManager] 图编译完成，新版本已生效。")

    def get_app(self):
        """获取当前活跃的应用实例"""
        self._compile_if_needed()
        return self.app

    def describe_main_path(self):
        if not self.entry_point:
            return "START"
        edges_map: Dict[str, List[str]] = {}
        for s, e in self.edges:
            edges_map.setdefault(s, []).append(e)
        path = [self.entry_point]
        visited = set()
        node = self.entry_point
        while node in edges_map and edges_map[node]:
            nxt = edges_map[node][0]
            if nxt in visited:
                break
            visited.add(nxt)
            path.append(nxt)
            node = nxt
        return "START -> " + " -> ".join(path)

# --- 动态节点实现 ---

def promotion_node(state: AgentState):
    """动态节点：促销检查"""
    print("  >>> [动态节点] PromotionNode正在运行...")
    # 模拟逻辑：如果查询了订单，追加一个优惠券
    return {"messages": [AIMessage(content="【系统提示】检测到您是尊贵会员，本单附赠一张 95 折优惠券！")]}

# --- 演示流程 ---

def run_demo():
    print("--- LangGraph 动态节点演示 (极简版) ---")

    manager = DynamicGraphManager()
    config = {"configurable": {"thread_id": "dynamic_demo"}}

    print(f"[节点顺序·添加前] {manager.describe_main_path()}")

    print("\n=== 阶段 2: 动态注入新节点 (运行时) ===")
    manager.register_node("promotion", promotion_node)
    manager.clear_edges_from("input_proc")
    manager.add_edge("input_proc", "promotion")
    manager.add_edge("promotion", "chatbot")
    print(f"[节点顺序·添加后] {manager.describe_main_path()}")

    current_app = manager.get_app()
    for event in current_app.stream({"messages": [HumanMessage(content="触发动态节点")]}, config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print(f"输出: {last_msg.content}")

if __name__ == "__main__":
    run_demo()
