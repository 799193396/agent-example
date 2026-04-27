from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen-plus-2025-04-28"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# LlamaIndex默认使用的大模型被替换为百炼
Settings.llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)

# 加载本地的嵌入模型
embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")
# 设置默认的向量模型为本地模型
Settings.embed_model = embed_model

# 定义本地化的向量化
chroma_client = chromadb.PersistentClient()
# get_or_create_collection:如果有本地实例化的chroma文件内容，直接获取对应的连接对象，否则创建新的连接对象
chroma_collection = chroma_client.get_or_create_collection("quickstart")
# 创建Chroma向量数据库对象
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 构建向量存储并自定义存储上下文
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# 加载文档并构建索引
documents = SimpleDirectoryReader(
    input_files=["data/deepseek介绍.txt"]
).load_data()

# 使用转换创建管道
# pipeline = IngestionPipeline(
#     transformations=[
#         SentenceSplitter(chunk_size=250, chunk_overlap=50),
#         TitleExtractor(),
#         embed_model,
#     ],
#     vector_store=vector_store
# )
#
# # 运行管道
# nodes = pipeline.run(documents=documents)
# 使用向量索引去进行存储(方式1)
index = VectorStoreIndex.from_documents(documents, show_progress=True, storage_context=storage_context)
# 可以使用摄取管道的方式去将向量存储和加载（方式2）
# index = VectorStoreIndex(nodes, show_progress=True, vector_store=vector_store)
print(index.as_retriever().retrieve("deepseek的公司收益？"))