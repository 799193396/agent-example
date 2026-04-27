from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')

DB_URI = "postgresql://postgres:123456@localhost:5432/postgres?sslmode=disable"

with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # 第一次使用postgres存储是需要调用初始化
    # store.setup()
    # checkpointer.setup()

    def call_model(
            state: MessagesState,
            config: RunnableConfig,
            *,
            store: BaseStore,
    ):
        # 获取对应的用户id
        user_id = config["configurable"]["user_id"]
        # 创建命名空间（唯一）
        namespace = ("memories", user_id)
        # 获取记忆
        memories = store.search(namespace, query=str(state["messages"][-1].content))
        info = "\n".join([d.value["data"] for d in memories])
        system_msg = f"你是一个与用户交谈的有帮助的助手. 用户信息: {info}"  # 构建历史消息
        # 和模型进行交互
        response = llm.invoke(
            [{"role": "system", "content": system_msg}] + state["messages"]  # 组装聊天历史
        )

        # 存储新的记忆，如果用户要求模型记住
        last_message = state["messages"][-1]
        if "记住" in last_message.content.lower():
            memory = "用户名是初见"
            # 存储记忆
            store.put(namespace, str(uuid.uuid4()), {"data": memory})

        return {"messages": response}

    # 创建图
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    config = {
        "configurable": {
            "thread_id": "1",
            "user_id": "1",
        }
    }
    for chunk in graph.stream(
            {"messages": [{"role": "user", "content": "嗨！记住：我的名字是初见"}]},
            config,
            stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()

    # 使用第二个线程去访问用户信息
    config = {
        "configurable": {
            "thread_id": "2",
            "user_id": "1",
        }
    }

    for chunk in graph.stream(
            {"messages": [{"role": "user", "content": "我的名字是什么？"}]},
            config,
            stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()