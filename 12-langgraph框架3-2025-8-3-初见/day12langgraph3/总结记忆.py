from typing import Any, TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')

# 创建一个专用于摘要的模型实例，限制输出最多 128 tokens
summarization_model = llm.bind(max_tokens=128)


# 定义状态结构，包含对话历史和摘要上下文
class State(MessagesState):
    context: dict[str, RunningSummary]  # 用于存储用户摘要记忆（running_summary）


# 定义输入格式，传给 call_model 函数使用
class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]  # 已被压缩/摘要过的消息
    context: dict[str, RunningSummary]


# 首次生成摘要
initial_summary_prompt = ChatPromptTemplate.from_template(
    """请阅读以下对话内容，并生成一个简洁的摘要，用于帮助理解对话的主要内容：

对话内容：
{messages}

摘要："""
)

# 在已有摘要基础上追加新的对话内容，更新摘要
existing_summary_prompt = ChatPromptTemplate.from_template(
    """你之前已经生成了如下摘要：
{existing_summary}

现在，对话继续发展了，请根据新增的对话内容，更新这个摘要，使其覆盖所有关键内容。

新增对话内容：
{messages}

更新后的摘要："""
)

# 用于最终调用模型之前，将摘要和剩余消息一起传入
final_prompt = ChatPromptTemplate.from_template(
    """你是一位智能助理。以下是用户和你的对话摘要，可帮助你快速理解上下文：

摘要：
{summary}

这是对话中未被总结的新消息，请继续处理这些信息：
{messages}
"""
)
# 创建摘要节点：超过一定 token 数时会对历史消息自动进行摘要
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,  # 使用近似 token 计算
    model=summarization_model,  # 使用绑定了 max_tokens 的模型
    max_tokens=200,  # 在进行摘要之前，传给模型的输入上下文的最大token长度限制
    max_tokens_before_summary=150,  # 超过这个数就会触发摘要
    max_summary_tokens=128,  # 每次摘要最多保留 128 tokens
    initial_summary_prompt=initial_summary_prompt,  # 首次生成摘要的提示词
    existing_summary_prompt=existing_summary_prompt,  # 更新摘要的提示词
    final_prompt=final_prompt  # 模型回答问题之前参考的摘要上下文的提示词
)


# 模型调用节点：对压缩过的历史消息进行问答
def call_llm(state: LLMInputState):
    response = llm.invoke(state["summarized_messages"])
    return {
        "messages": [response],
        "context": state.get("context", {})  # 把上下文原样返回，里面就有摘要
    }


# 使用内存存储器（可换成 Redis/Postgres）
checkpointer = InMemorySaver()

# 构建 LangGraph 的流程图
builder = StateGraph(State)

# 添加两个节点：摘要节点 和 模型调用节点
builder.add_node(call_llm)
builder.add_node("summarize", summarization_node)

# 定义边：从 START 开始 → 先摘要 → 再模型调用
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_llm")

# 编译图
graph = builder.compile(checkpointer=checkpointer)

# ========== 流程调用 ==========
config = {"configurable": {"thread_id": "1"}}  # 每个线程维护一个上下文

# 第1轮：告诉模型「我叫小明」
graph.invoke({"messages": "你好，我叫初见"}, config)

# 第2轮：要求写一首猫的诗
graph.invoke({"messages": "请写一首关于猫的诗"}, config)

# 第3轮：让它对狗做一样的事
graph.invoke({"messages": "现在也请为狗写一首诗"}, config)

# 第4轮：问它「我叫什么名字？」
final_response = graph.invoke({"messages": "你还记得我叫什么名字吗？"}, config)

# 输出最终回复
final_response["messages"][-1].pretty_print()

# 输出摘要内容（短期记忆）
print("\n摘要记忆内容（summary）:", final_response)