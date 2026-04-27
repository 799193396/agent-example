from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')


def call_llm(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",  # 修剪策略（last从末尾，first从开头， middle从中间）
        token_counter=count_tokens_approximately,  # 用来估算token数量
        max_tokens=50,  # 修剪后的消息总 token 不超过 200
        start_on="human",  # 控制从哪一类消息开始截取（从最后一个 human 消息开始往前保留）
        end_on=("human", "tool"),  # 允许哪些角色作为修剪终点（修剪过程中只能包括 "human" 和 "tool" 类型的消息）
    )
    print(f"修剪后消息数量: {len(messages)}")
    print(f"修剪后总tokens: {count_tokens_approximately(messages)}")
    print("修剪后的消息:")
    for i, msg in enumerate(messages):
        print(f"  {i}: {msg.type} - {msg.content[:50]}...")
    print("\n" + "=" * 50 + "\n")

    response = llm.invoke(messages)
    return {"messages": [response]}


checkpointer = InMemorySaver()
builder = StateGraph(MessagesState)
builder.add_node(call_llm)
builder.add_edge(START, "call_llm")
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "我的名字叫初见"}, config)
graph.invoke({"messages": "帮我家的猫写一首诗"}, config)
graph.invoke({"messages": "现在对狗做一样的事情"}, config)
final_response = graph.invoke({"messages": "我的名字叫什么?"}, config)

# print(final_response)
# final_response["messages"][-1].pretty_print()
