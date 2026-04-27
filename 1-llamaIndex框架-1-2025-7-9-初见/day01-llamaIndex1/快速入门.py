from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen-turbo"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True, max_tokens=1000)
#
# response = llm.complete("帮我推荐一下江浙沪5天的旅游攻略")
# print(response)


# LlamaIndex默认使用的大模型被替换为百炼
Settings.llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
# 加载本地的嵌入模型
Settings.embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")

# 从文件目录加载文件,自动选择对应的文档加载器
documents = SimpleDirectoryReader("data").load_data()
# 从文档创建索引
index = VectorStoreIndex.from_documents(documents)
# 将索引转换为查询引擎
query_engine = index.as_query_engine()
response = query_engine.query("企业事件？")
print(response)
