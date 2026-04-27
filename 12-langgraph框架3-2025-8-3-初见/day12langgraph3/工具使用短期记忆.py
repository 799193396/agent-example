from typing import Annotated, NotRequired
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')


class CustomState(AgentState):
    # user_name字段处于短期状态
    user_name: NotRequired[str]


# 只有通过 InjectedState，LangGraph 才会自动把当前运行状态（CustomState）注入到工具函数中。
@tool
def get_user_name(
        state: Annotated[CustomState, InjectedState]
) -> str:
    """从state中检索当前用户名。"""
    # 返回存储的名称，如果未设置则返回默认值
    return state.get("user_name", "Unknown user")


@tool
def update_user_name(
        new_name: str,
        # 注入当前工具调用ID（LLM会忽略此参数，用于工具消息追踪）
        tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """更新短期记忆中的用户名"""
    return Command(update={
        "user_name": new_name,
        "messages": [
            ToolMessage(content=f"名字已更新为：{new_name}", tool_call_id=tool_call_id)
        ]
    })


# 创建内存持久化器
checkpointer = InMemorySaver()

# 创建代理----agent可以直接作为工作流的节点
agent = create_react_agent(
    model=llm,
    tools=[get_user_name, update_user_name],
    state_schema=CustomState,
    checkpointer=checkpointer,
)
config = {"configurable": {"thread_id": "1"}}

# 第一次：更新用户名（模型会调用update_user_name工具）
response = agent.invoke({"messages": [{"role": "user", "content": "我的名字是初见"}]}, config)
response["messages"][-1].pretty_print()

# 第二次：获取用户名（调用get_user_name工具）
response = agent.invoke({"messages": [{"role": "user", "content": "我的名字是什么?"}]}, config)
response["messages"][-1].pretty_print()