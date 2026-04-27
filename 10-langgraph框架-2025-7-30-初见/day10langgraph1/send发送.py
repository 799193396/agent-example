from langgraph.graph import StateGraph, END
from langgraph.types import Send
from typing import TypedDict

class MyState(TypedDict):
    messages: list
    result: str

def process_input(state: MyState) -> Send:
    """处理输入并决定发送到哪个节点"""
    if len(state["messages"]) > 5:
        return Send("complex_processor", state)
    else:
        return Send("simple_processor", state)

def simple_processor(state: MyState) -> MyState:
    """简单处理器"""
    return {"messages": state["messages"], "result": "simple"}

def complex_processor(state: MyState) -> MyState:
    """复杂处理器"""
    return {"messages": state["messages"], "result": "complex"}

# 构建图
graph = StateGraph(state_schema=MyState)
graph.add_node("input", process_input)
graph.add_node("simple_processor", simple_processor)
graph.add_node("complex_processor", complex_processor)

graph.set_entry_point("input")
graph.add_edge("simple_processor", END)
graph.add_edge("complex_processor", END)
app = graph.compile()

# 定义数据
massages = []
for i in range(3):
    massages.append(i)
print(app.invoke({"messages": massages}))