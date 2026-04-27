from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')


# 创建子图
def subplot(state: MessagesState) -> MessagesState:
    # 获取大模型回答的内容进行摘要总结
    answer = state["messages"][-1].content
    summary_prompt = f"请用一句话总结下面这句话：\n\n答：{answer}"
    response = llm.invoke(summary_prompt)
    return {"messages": state["messages"] + [response]}


summary_subgraph = (
    StateGraph(state_schema=MessagesState)
    .add_node("subplot", subplot)
    .add_edge(START, "subplot")
    .compile()
)


# 创建父图
def llm_answer_node(state: MessagesState) -> MessagesState:
    # 使用大模型进行回答
    answer = llm.invoke(state["messages"])
    print("父图输出", answer)
    return {"messages": state["messages"] + [answer]}


checkpointer = InMemorySaver()
parent_graph = (
    StateGraph(MessagesState)
    .add_node("llm_answer", llm_answer_node)
    .add_node("summarize_subgraph", summary_subgraph)
    .add_edge(START, "llm_answer")
    .add_edge("llm_answer", "summarize_subgraph")
    .compile(checkpointer=checkpointer)
)
config = {"configurable": {"thread_id": "1"}}
# 测试输入
input_state = {
    "messages": [{"role": "user", "content": "langgraph是什么？请用100字介绍"}],
}

for chunk in parent_graph.stream(
        input_state,
        config,
        stream_mode="updates",
        subgraphs=True

):
    # print(chunk)
    pass
print("================获取状态=================")
"""
与已保存的图表状态交互时，必须指定线程标识符。您可以通过调用来查看图表的最新graph.get_state(config)状态。这将返回一个StateSnapshot对象
"""
print(parent_graph.get_state(config).values)

print("================状态历史记录=================")
"""
可以通过调用 获取给定线程的图形执行的完整历史记录graph.get_state_history(config)。
这将返回与配置中提供的线程 ID 关联的对象列表StateSnapshot。
重要的是，检查点将按时间顺序排序，最新的检查点 /StateSnapshot将位于列表中的第一个。

注意：这里采用的是共享状态的子图，可以将子图的内容持久化，如果使用的是不同状态的就需要分别存储
"""
history = list(parent_graph.get_state_history(config))
for idx, snapshot in enumerate(history):
    print(f"Step {idx}:")
    print(f"  Checkpoint ID: {snapshot.config['configurable']['checkpoint_id']}")
    print(f"  Node: {snapshot.metadata.get('source')}")
    print(f"  Messages: {[m.content for m in snapshot.values['messages']]}")
    print("")

print("================重放机制=================")
"""
可以重放先前的图执行。如果使用 thread_id 和 checkpoint_id 调用图，LangGraph 会：
    1.重放检查点之前已执行的步骤（不重新执行）
    2.执行检查点之后的步骤（即使之前已执行）
注意：必须传递这些内容thread_id， checkpoint_id

重要特性：
    LangGraph 知道特定步骤是否已执行
    检查点前的步骤会被重放（不重新执行）
    检查点后的步骤会被重新执行，不包括当前检查点（创建新分支）
"""
# 获取Step2检查点，开始进行重放
step2_level_checkpoint = None
if history:
    step2_level_checkpoint = list(history)[2].config['configurable']['checkpoint_id']

print("第二步检查点：", step2_level_checkpoint)
config = {"configurable": {"thread_id": "1", "checkpoint_id": step2_level_checkpoint}}
# 开始执行
result = parent_graph.invoke(None, config=config)
print(result)