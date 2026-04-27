from typing import Annotated, List
from typing_extensions import TypedDict
from operator import add
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化大模型
llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')


# 定义状态结构
class MessagesState(TypedDict):
    messages: Annotated[List[BaseMessage], add]


# 创建检查点保存器和内存存储
checkpointer = InMemorySaver()
in_memory_store = InMemoryStore()


# 聊天机器人节点  *代表后面的参数必须使用显示写出参数名称  store=in_memory_store
def chatbot(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """主聊天机器人节点，处理用户消息并生成回复"""

    # 获取用户ID和最新消息
    user_id = config["configurable"]["user_id"]
    last_message = state["messages"][-1]

    # 定义内存命名空间
    namespace = (user_id, "memories")

    # 简单的聊天逻辑
    user_input = last_message.content.lower()
    # 将聊天历史获取并组装提示词
    memories = store.search(namespace)
    memory_text = "\n".join(m.value["memory"] for m in memories)
    prompt = f"请参考聊天记录：{memory_text}\n\nHuman: {user_input}\nAI:"

    response = llm.invoke(prompt).content

    # 存储对应的问题和答案
    memory = f"问题：{user_input} --- 答案:{response}"
    memory_id = str(uuid.uuid4())
    store.put(namespace, memory_id, {"memory": memory})
    # 返回AI消息
    return {"messages": [AIMessage(content=response)]}


# 创建图
def create_persistent_graph():
    """创建持久化的聊天机器人图"""

    # 创建状态图
    workflow = StateGraph(MessagesState)
    # 添加节点
    workflow.add_node("chatbot", chatbot)
    # 添加边
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)

    # 编译图，使用检查点保存器和存储
    graph = workflow.compile(checkpointer=checkpointer, store=in_memory_store)

    return graph


# 工具函数：显示状态历史
def show_state_history(graph, config):
    """显示状态历史"""
    print("\n=== 状态历史 ===")
    history = graph.get_state_history(config)
    for i, snapshot in enumerate(history):
        print(f"\n步骤 {i}:")
        print(f"  配置: {snapshot.config}")
        print(f"  值: {snapshot.values}")
        print(f"  下一步: {snapshot.next}")
        print(f"  元数据: {snapshot.metadata}")


# 工具函数：显示存储的记忆
def show_memories(store, user_id):
    """显示用户的所有记忆"""
    print(f"\n=== 用户 {user_id} 的记忆 ===")
    namespace = (user_id, "memories")
    memories = store.search(namespace)

    if memories:
        for memory in memories:
            print(f"记忆ID: {memory.key}")
            print(f"内容: {memory.value}")
            print(f"创建时间: {memory.created_at}")
            print(f"更新时间: {memory.updated_at}")
            print("---")
    else:
        print("没有找到记忆")


# 主程序
def main():
    # 创建图
    graph = create_persistent_graph()

    # 用户配置
    user_id = "user_123"
    thread_id = "conversation_1"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }

    print("=== LangGraph 持久化聊天机器人 ===")
    print("输入 'quit' 退出，'history' 查看状态历史，'memories' 查看记忆")

    while True:
        user_input = input("\n用户: ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'history':
            show_state_history(graph, config)
            continue
        elif user_input.lower() == 'memories':
            show_memories(in_memory_store, user_id)
            continue

        # 创建用户消息
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }

        # 运行图
        try:
            result = graph.invoke(initial_state, config)

            # 显示AI回复
            ai_message = result["messages"][-1]
            print(f"AI: {ai_message.content}")

        except Exception as e:
            print(f"错误: {e}")

    print("\n=== 最终状态 ===")
    final_state = graph.get_state(config)
    print(f"最终状态: {final_state.values}")

    print("\n=== 所有记忆 ===")
    show_memories(in_memory_store, user_id)


# 演示不同线程间的记忆共享
def demo_cross_thread_memory():
    """演示跨线程记忆共享"""
    print("\n=== 跨线程记忆共享演示 ===")

    graph = create_persistent_graph()
    user_id = "user_456"

    # 第一个对话线程
    config1 = {
        "configurable": {
            "thread_id": "thread_1",
            "user_id": user_id
        }
    }

    print("线程1 - 建立记忆:")
    result1 = graph.invoke({
        "messages": [HumanMessage(content="我叫Alice，我喜欢音乐")]
    }, config1)
    print(f"AI: {result1['messages'][-1].content}")

    # 第二个对话线程（相同用户）
    config2 = {
        "configurable": {
            "thread_id": "thread_2",
            "user_id": user_id
        }
    }

    print("\n线程2 - 访问记忆:")
    result2 = graph.invoke({
        "messages": [HumanMessage(content="你还记得我吗？")]
    }, config2)
    print(f"AI: {result2['messages'][-1].content}")

    # 显示共享的记忆
    show_memories(in_memory_store, user_id)


if __name__ == "__main__":
    # 运行主程序
    main()

    # 演示跨线程记忆共享
    demo_cross_thread_memory()
