from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from dotenv import load_dotenv
import os
load_dotenv()

model = "qwen-plus-2025-07-14"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")
llm = ChatOpenAI(model=model, api_key=api_key, base_url=api_base_url)

# 创建搜索工具
search = SerpAPIWrapper()

# 定义搜索工具
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="用于搜索问题的中间答案。输入应该是一个搜索查询。"
    )
]

# 创建Self-Ask代理
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True,
    handle_parsing_errors=True
)

# 测试代理
question = "2024年巴黎奥运会中国获得了多少枚金牌？中国在奖牌榜上排第几？"
result = agent.run(question)
print(f"代理最终答案: {result}")