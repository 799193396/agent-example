from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """将两个数目相乘."""
    return a * b


# 运行工具
print(multiply.invoke({"a": 6, "b": 7}))  # returns 42

tool_call = {
    "type": "tool_call",
    "id": "1",
    "args": {"a": 42, "b": 7}
}
# print(multiply.invoke(tool_call))

print("=" * 8, "在代理中使用", "=" * 8)
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')

agent = create_react_agent(
    model=llm,
    tools=[multiply]
)
# print(agent.invoke({"messages": [{"role": "user", "content": "100乘以37等于多少？"}]}))

print("=" * 8, "在工作流中使用", "=" * 8)
"""function calling"""
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_tavily import TavilySearch


@tool
def tavily_search_tool(query: str) -> str:
    """这是一个搜索工具"""
    # 可以在搜索工具执行之前对问题进行操作（生成类似问题），对搜索的结果进行过滤，清洗
    tool_instance = TavilySearch()
    return tool_instance.run(query)

# 执行工具的节点
tool_node = ToolNode([tavily_search_tool])
# 绑定工具到模型
model_with_tools = llm.bind_tools([tavily_search_tool])


# 条件函数
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)

# 定义节点和边
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", "call_model")

graph = builder.compile()

print(graph.invoke({"messages": [{"role": "user", "content": "上海的天气?"}]}))