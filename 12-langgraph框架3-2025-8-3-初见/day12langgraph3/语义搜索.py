from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.graph import START, MessagesState, StateGraph
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')

# 加载嵌入模型
embeddings = HuggingFaceEmbeddings(model_name=r"D:\llm\Local_model\BAAI\bge-large-zh-v1___5")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1024,
    }
)

store.put(("memories", "user_123"), "1", {"text": "我喜欢吃披萨"})
store.put(("memories", "user_123"), "2", {"text": "我喜欢吃红烧肉"})
store.put(("memories", "user_123"), "3", {"text": "我的职业是程序员"})


def chat(state, *, store: BaseStore):
    # 根据用户的最后一条消息进行搜索
    items = store.search(
        ("memories", "user_123"), query=state["messages"][-1].content, limit=2
    )
    print(items)
    memories = "\n".join(item.value["text"] for item in items)
    memories = f"## 用户记忆\n{memories}" if memories else ""
    response = llm.invoke(
        [
            {"role": "system", "content": f"你是一个乐于助人的助手.\n{memories}"},

        ] + state["messages"]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node(chat)
builder.add_edge(START, "chat")
graph = builder.compile(store=store)

for message, metadata in graph.stream(
        input={"messages": [{"role": "user", "content": "我饿了？"}]},
        stream_mode="messages",
):
    print(message.content, end="")