# 导入必要的库
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import numexpr
import math
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 定义模型参数
model = "qwen-turbo-latest"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# 初始化LLM
llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=api_base_url
)

# 构建一个搜索工具
search = TavilySearch()

# 创建一个数学计算工具
@tool
def calculator(expression: str) -> str:
    """数学计算器"""
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        result = numexpr.evaluate(expression.strip(), global_dict={}, local_dict=local_dict)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

# 定义工具列表（直接使用现成的工具）
tools = [
    search,
    calculator
]

# 记忆组件
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 使用最新的Agent创建方式
custom_prompt = PromptTemplate.from_template("""
你是一个有用的AI助手，能够使用工具来回答问题。你有以下工具可用：

{tools}

使用以下格式：

Question: 需要回答的输入问题
Thought: 你应该总是思考该做什么
Action: 要采取的行动，应该是[{tool_names}]中的一个
Action Input: 行动的输入
Observation: 行动的结果
... (这个Thought/Action/Action Input/Observation可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 对原始输入问题的最终答案

重要提示：
1. 如果聊天历史中有相关信息，请参考并利用这些信息
2. 聊天历史记录了之前的对话内容，你应该能够回忆和引用

聊天历史:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
""")

# 创建Agent
agent = create_react_agent(
    llm,
    tools,
    custom_prompt
)
# 创建Agent管理器
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               memory=memory,
                               verbose=True)

# 测试对话
print("=== 第一次对话 ===")
response1 = agent_executor.invoke({"input": "请问2025年的美国总统是谁？"})
print(f"回答: {response1['output']}")

print("\n=== 第二次对话 ===")
response2 = agent_executor.invoke({"input": "好厉害，刚才我们都聊了什么？"})
print(f"回答: {response2['output']}")

print("\n=== 第三次对话（测试记忆） ===")
response3 = agent_executor.invoke({"input": "请计算 2+3*4，然后告诉我你还记得我们之前讨论过什么话题？"})
print(f"回答: {response3['output']}")