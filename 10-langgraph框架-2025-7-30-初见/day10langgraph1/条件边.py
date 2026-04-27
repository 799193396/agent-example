from typing import TypedDict
from langgraph.graph import StateGraph, END


class MyState(TypedDict):
    type: str
    result: str


def judge_node(state: MyState):
    """节点函数：可以做一些预处理"""
    return state  # 保持状态不变，只是路由


def route_condition(state: MyState):
    """条件函数：只负责路由决策"""
    if state["type"] == "a":
        return "a"
    elif state["type"] == "b":
        return "b"
    else:
        return "default"


def node_a(state):
    return {"result": "走了 A 分支"}


def node_b(state):
    return {"result": "走了 B 分支"}


def node_default(state):
    return {"result": "走了默认分支"}


# 构建图
graph = StateGraph(state_schema=MyState)
# 定义节点
graph.add_node("judge_node", judge_node)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_node("default", node_default)
# 定义开始节点
graph.set_entry_point("judge_node")

# 使用不同的函数作为条件函数，
# 参数1：从哪个节点结束后执行该条件，参数2：条件判断的函数
# 参数3：该去往哪一个节点（根据value）
graph.add_conditional_edges("judge_node", route_condition, {
    "a": "b",
    "b": "a",
    "default": "default"
})

# 添加结束边
graph.add_edge("a", END)
graph.add_edge("b", END)
graph.add_edge("default", END)

app = graph.compile()

# 测试
print("测试 A:", app.invoke({"type": "a", "result": ""}))