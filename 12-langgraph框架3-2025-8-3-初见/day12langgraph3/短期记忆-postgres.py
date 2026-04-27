from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')

DB_URI = "postgresql://postgres:123456@localhost:5432/postgres?sslmode=disable"
# with管理资源的获取和释放
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()  # 执行该代码的时候才会创建对应的表

    def call_llm(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}


    builder = StateGraph(MessagesState)
    builder.add_node(call_llm)
    builder.add_edge(START, "call_llm")

    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    chunk = graph.invoke(
        {"messages": [{"role": "user", "content": "你好，我叫法外狂徒张三"}]},
        config,
        stream_mode="values"
    )
    chunk["messages"][-1].pretty_print()

    chunk1 = graph.invoke(
        {"messages": [{"role": "user", "content": "我的名字叫什么？"}]},
        config,
        stream_mode="values"
    )
    chunk1["messages"][-1].pretty_print()

    # 最终状态检查
    final_state = graph.get_state(config)
    print(f"\n=== 最终状态 ===")
    print(f"总消息数: {len(final_state.values.get('messages', []))}")

    for i, msg in enumerate(final_state.values.get('messages', [])):
        role = getattr(msg, 'type', 'unknown')
        content = getattr(msg, 'content', str(msg))
        print(f"  {i + 1}. [{role}] {content[:100]}...")