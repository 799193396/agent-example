from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import Command, Literal, Send


class MyState(TypedDict):
    type: str
    text: str
    result: str


def judge_node(state: MyState) -> Command[Literal["a", "b", "default"]]:
    """条件函数：使用Command进行路由和状态更新"""
    if state["type"] == "a":
        return Command(update={"text": "走了 A 分支", "name": "张三"}, goto="a")
    elif state["type"] == "b":
        # Command只是更新当前节点结束的状态
        # return Command(update={"text": "走了 B 分支"}, goto="b")
        return Send("b", state)
    else:
        return Command(update={"text": "走了默认分支"}, goto="default")


def node_a(state):
    print(state["name"])
    return {"result": f"A节点处理: {state['text']}"}


def node_b(state):
    return {"result": f"B节点处理: {state['text']}"}


def node_default(state):
    return {"result": f"默认节点处理: {state['text']}"}


# 构建图
graph = StateGraph(state_schema=MyState)
graph.add_node("judge_node", judge_node)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_node("default", node_default)

graph.set_entry_point("judge_node")

# 添加结束边
graph.add_edge("a", END)
graph.add_edge("b", END)
graph.add_edge("default", END)

app = graph.compile()

# 测试
print("测试 A:", app.invoke({"type": "a", "text": "", "result": ""}))
print("测试 B:", app.invoke({"type": "b", "text": "", "result": ""}))
print("测试其他:", app.invoke({"type": "default", "text": "", "result": ""}))