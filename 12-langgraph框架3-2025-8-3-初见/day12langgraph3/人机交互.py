from typing import TypedDict
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph

from langgraph.types import interrupt, Command


# 1. 定义状态结构，LangGraph 的每个节点输入输出都以它为标准
class State(TypedDict):
    some_text: str  # 保存一段文本


# 2. 定义一个“人工参与节点”
def human_node(state: State):
    # 使用 interrupt 让图在这里暂停，并把 current state 输出出去， value是用户选择的内容
    value = interrupt(
        {
            "是否同意进行下一步": "1.同意，2.拒绝"  # 提示要修改的文本
        }
    )
    if not value:
        return {
            "some_text": "拒绝请假"  # 更新状态为人工输入后的新文本
        }
    # interrupt 实际上在 resume 之后会把 resume 的内容作为 value
    return {
        "some_text": "成功请假"  # 更新状态为人工输入后的新文本
    }


# 3. 构建图
graph_builder = StateGraph(State)  # 指定状态类型
graph_builder.add_node("human_node", human_node)  # 添加节点
graph_builder.add_edge(START, "human_node")  # 从起点进入这个节点

# 4. 启用内存检查点（用于保存每个线程的执行状态）
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)  # 编译成可执行图

# 5. 设置一个线程ID，便于在中断后恢复同一个会话
config = {"configurable": {"thread_id": uuid.uuid4()}}

# 6. 启动流程，并传入初始文本。运行到中断节点时会暂停
result = graph.invoke({"some_text": "original text", "name": "张三"}, config=config)

# 7. 输出中断返回内容，通常是交给用户编辑或确认
print(result["__interrupt__"])
# 例子输出：{'name': 'human_node', 'args': {'text_to_revise': 'original text'}}


# 8. 模拟用户修改后，通过 resume 恢复图的执行（传入 Command）
print(graph.invoke(Command(resume=False), config=config))

# 输出：{'some_text': 'Edited text'} 表示人工干预后的新状态
