from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm(model: str = "qwen-turbo"):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    api_base_url = os.getenv("DASHSCOPE_BASE_URL")

    # LlamaIndex默认使用的大模型被替换为百炼
    llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
    Settings.llm = llm

    # 加载本地的嵌入模型
    embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")
    # 设置默认的向量模型为本地模型
    Settings.embed_model = embed_model

    return llm, embed_model
