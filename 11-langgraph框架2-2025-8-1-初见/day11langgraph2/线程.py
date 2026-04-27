from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')


def process_message(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(state_schema=MessagesState)
builder.add_node("process_message", process_message)
builder.add_edge(START, "process_message")
# # 没有使用持久性
# graph = builder.compile()
#
# input_message = {"role": "user", "content": "你好呀！我的名字叫初见"}
# for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
#     chunk["messages"][-1].pretty_print()
#
# input_message = {"role": "user", "content": "我的名字叫什么？"}
# for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
#     chunk["messages"][-1].pretty_print()

# 使用持久性
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "user_123"}}

input_message = {"role": "user", "content": "你好呀！我的名字叫初见"}
for chunk in graph.stream({"messages": [input_message]}, config=config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_message = {"role": "user", "content": "我的名字叫什么？"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()