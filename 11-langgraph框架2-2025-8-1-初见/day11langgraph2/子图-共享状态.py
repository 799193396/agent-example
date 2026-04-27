from langgraph.graph import StateGraph, MessagesState, START
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


parent_graph = (
    StateGraph(MessagesState)
    .add_node("llm_answer", llm_answer_node)
    .add_node("summarize_subgraph", summary_subgraph)
    .add_edge(START, "llm_answer")
    .add_edge("llm_answer", "summarize_subgraph")
    .compile()
)

# 测试
input_state = {
    "messages": [{"role": "user", "content": "langgraph是什么？"}],
}
result = parent_graph.invoke(input_state)
print(result)