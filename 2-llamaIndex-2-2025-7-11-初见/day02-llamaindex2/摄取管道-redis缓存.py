from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen-max-latest"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# LlamaIndex默认使用的大模型被替换为百炼
Settings.llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)

# 加载本地的嵌入模型
embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")
# 设置默认的向量模型为本地模型
Settings.embed_model = embed_model

# 定义数据连接器去读取数据
documents = SimpleDirectoryReader(input_files=["data/小说.txt"]).load_data()

ingest_cache = IngestionCache(
    cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
    collection="my_test_cache",
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=250, chunk_overlap=50),
        TitleExtractor(),
        embed_model,
    ],
    cache=ingest_cache,
)

# 直接将数据摄取到向量数据库
pipeline.run(documents=documents)

# 加载和恢复状态
new_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=250, chunk_overlap=50),
        TitleExtractor(),
        embed_model,
    ]
)

# 由于缓存的存在会立即执行
nodes = pipeline.run(documents=documents)

print(nodes)
for node in nodes:
    print(node, "\n\n")