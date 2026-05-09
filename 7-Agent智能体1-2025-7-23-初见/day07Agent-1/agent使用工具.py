from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub  # 使用线上提示词
from langchain_tavily import TavilySearch  # https://app.tavily.com/
from dotenv import load_dotenv
import os

load_dotenv()

model = "qwen-turbo-latest"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")


# 定义llm
llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=api_base_url
)

# 工具列表
tools = [
    TavilySearch(),
]

# 使用新的Agent创建方式
prompt = hub.pull("hwchase17/react")  # 获取线上react提示词

agent = create_react_agent(
    llm,
    tools,
    prompt
)
# Agent管理器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "请问现任的美国总统是谁？他的年龄的平方是多少? 请用中文告诉我这两个问题的答案"}))