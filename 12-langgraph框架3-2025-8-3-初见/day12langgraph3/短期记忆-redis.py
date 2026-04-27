from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.redis import RedisSaver
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')

DB_URI = "redis://localhost:6379"
with RedisSaver.from_conn_string(DB_URI) as checkpointer:
    # 第一次使用 Redis 检查点时需要调用，在第一个创建表的时候使用
    # checkpointer.setup()


    # 执行模型对话的函数图
    def call_llm(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": response}


    # 创建图对象
    builder = StateGraph(MessagesState)
    builder.add_node(call_llm)
    builder.add_edge(START, "call_llm")

    # 将检查点传入
    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    # 第一次对话
    for chunk in graph.stream(
            {"messages": [{"role": "user", "content": "你好呀，我是初见"}]},
            config,
            stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    # 第二次对话
    for chunk in graph.stream(
            {"messages": [{"role": "user", "content": "我的名字叫什么?"}]},
            {
                "configurable": {
                    "thread_id": "20"
                }
            },
            stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()