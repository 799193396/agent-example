from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')  # 模型最好是使用最新发布的模型


# 定义工具函数
@tool
def search_web(query: str) -> str:
    """搜索网络信息的工具"""
    t_search = TavilySearchResults()

    return t_search.run(query)


@tool
def get_data_tool():
    """获取目前日期的工具"""
    return datetime.now().date()


tools = [search_web, get_data_tool]

system_prompt = """你是一个智能助手。你有以下工具可以使用：

1. search_web: 用于搜索互联网获取最新信息，特别是产品价格、新闻、实时数据等
2. get_current_date: 获取今天的日期

重要规则：
- 当用户询问产品价格、最新信息、新闻等需要实时数据的问题时，必须使用search_web工具
- 当用户询问时间或日期时，使用相应的时间工具
- 如果你的知识库中没有准确或最新的信息，应该使用搜索工具
- 优先使用工具获取准确信息，而不是依赖可能过时的训练数据

请根据用户问题选择合适的工具来获取准确答案。"""

agent = create_react_agent(model=llm,
                           tools=tools,
                           prompt=system_prompt
                           )

response = agent.invoke({"messages": [{"role": "user", "content": "请帮我搜索一下特斯拉的最新消息？"}]})

print(response)