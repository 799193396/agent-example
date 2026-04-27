from typing import TypedDict
from langgraph.graph import StateGraph
from typing import Annotated
import operator


# 定义状态结构   如果定义的是list[dict]，会覆盖之前的数据
class ChatState(TypedDict):
    messages: Annotated[list, operator.add]  # 每条消息是 {role, content}，会自动追加到列表末尾


# 节点函数：添加用户问题
def user_input_node(state: ChatState) -> dict:
    user_msg = {"role": "user", "content": "什么是LangGraph？"}
    return {"messages": [user_msg]}


# 节点函数：添加助手回复
def assistant_node(state: ChatState) -> dict:
    reply = {"role": "assistant", "content": "LangGraph 是一个有状态的图编排框架。"}
    return {"messages": [reply]}


# 构建状态图
builder = StateGraph(state_schema=ChatState)
builder.add_node("user_input", user_input_node)
builder.add_node("assistant_reply", assistant_node)
builder.set_entry_point("user_input")  # 设置开始节点
builder.add_edge("user_input", "assistant_reply")
# 编译图
graph = builder.compile()

result = graph.invoke({"messages": []})
print(result["messages"])