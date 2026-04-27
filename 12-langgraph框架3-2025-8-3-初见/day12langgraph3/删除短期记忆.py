from langchain_core.messages import RemoveMessage
from langgraph.graph import StateGraph, START, MessagesState
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')


def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # 删除最早的两条消息
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None


def call_llm(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_sequence([call_llm, delete_messages])
builder.add_edge(START, "call_llm")

checkpointer = InMemorySaver()
app = builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

for event in app.stream(
        {"messages": [{"role": "user", "content": "你好呀，我是初见哦"}]},
        config,
        stream_mode="values"
):
    print([(message.type, message.content) for message in event["messages"]])

for event in app.stream(
        {"messages": [{"role": "user", "content": "我的名字是什么？"}]},
        config,
        stream_mode="values"
):
    # 最终回复会把最开始的两条消息删除
    print([(message.type, message.content) for message in event["messages"]])