from langgraph.graph import StateGraph, MessagesState, START
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')


# 创建子图
class SubgraphMessagesState(TypedDict):
    subgraph_messages: Annotated[list[AnyMessage], add_messages]


def subplot(state: SubgraphMessagesState) -> SubgraphMessagesState:
    # 获取大模型回答的内容进行摘要总结
    answer = state["subgraph_messages"][-1].content
    summary_prompt = f"请用一句话总结下面这句话：\n\n答：{answer}"
    response = llm.invoke(summary_prompt)
    # print("\n\n")
    # print("子图中问题和输出:", state["subgraph_messages"] + [response])
    return {"subgraph_messages": [response]}


summary_subgraph = (
    StateGraph(state_schema=SubgraphMessagesState)
    .add_node("subplot", subplot)
    .add_edge(START, "subplot")
    .compile()
)


# 创建父图

def llm_answer_node(state: MessagesState) -> MessagesState:
    # 使用大模型进行回答
    answer = llm.invoke(state["messages"])
    # print("父图中问题和输出:", state["messages"] + [answer])
    # 转换状态格式
    summary_result = summary_subgraph.invoke({"subgraph_messages": state["messages"] + [answer]})

    return {"messages": state["messages"] + [answer, summary_result["subgraph_messages"][2]]}


parent_graph = (
    StateGraph(state_schema=MessagesState)
    .add_node("llm_answer", llm_answer_node)
    .add_edge(START, "llm_answer")
    .compile()
)
# 测试输入
input_state = {
    "messages": [{"role": "user", "content": "langgraph是什么？"}],
}

for chunk in parent_graph.stream(
        input_state,
        stream_mode="updates",
        subgraphs=True,
):
    print(chunk)