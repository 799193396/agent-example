from typing import TypedDict
from langgraph.graph import StateGraph

# 定义状态结构
class MyState(TypedDict):
    question: str
    answer: str

# 定义节点函数
def search_node(state):
    print(state["question"])
    return {"answer": "这是答案"}

# 创建状态图
builder = StateGraph(state_schema=MyState)
# 添加一个节点
builder.add_node("search", search_node)
# 第一个要调用的节点， 开始节点
builder.set_entry_point("search")
# 要构建图，首先要定义状态，然后添加节点和边，最后进行编译，会进行基本的检查
graph = builder.compile()

# 执行图
result = graph.invoke({"question": "什么是状态图？"})
print(result["answer"])  # 输出：这是答案