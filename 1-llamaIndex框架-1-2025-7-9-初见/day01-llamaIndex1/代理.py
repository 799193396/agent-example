import asyncio
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen-max-2025-01-25"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")


# 定义一个简单的计算器工具
def multiply(a: float, b: float) -> float:
    """
    简单计算器
    Args:
        a: 第一个数字
        b: 第二个数字
        operation: 运算符
    Returns:
        计算结果
    :param a:
    :param b:
    :return:
    """
    return a * b

# 将函数封装成 LlamaIndex 的工具
multiply_tool = FunctionTool.from_defaults(fn=multiply)
tools = [multiply_tool]
# 创建一个代理工作流，将计算器工具传入
agent = ReActAgent.from_tools(
    tools=tools,
    llm=DashScope(model_name=model, api_key=api_key),
    verbose=True
)


async def main():
    # 运行代理
    response = await agent.achat("2乘以2等于多少?")
    print(str(response))


# 运行代理
if __name__ == "__main__":
    asyncio.run(main())