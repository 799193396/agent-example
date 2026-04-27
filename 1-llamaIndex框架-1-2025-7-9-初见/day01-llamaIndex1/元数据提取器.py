from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core.ingestion import IngestionPipeline  # 创建摄取管道
from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen-max-latest"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# LlamaIndex默认使用的大模型被替换为百炼
Settings.llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
# 加载本地的嵌入模型
Settings.embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")

documents = SimpleDirectoryReader("data").load_data()

# 分割文本设置
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)
# 提取每个节点上下文的标题-根据几个node节点生成标题
title_extractor = TitleExtractor(nodes=5)
# 为每一个节点生成问题
qa_extractor = QuestionsAnsweredExtractor(questions=3)

# 将原始数据转换为可用于查询的结构化格式
pipeline = IngestionPipeline(
    transformations=[text_splitter, title_extractor, qa_extractor]
)
# 开始执行将原始数据转换为可索引的文档格式
nodes = pipeline.run(
    documents=documents,
    in_place=True,
    show_progress=True,
)
print(nodes)

# 或者直接插入到索引中
# from llama_index.core import VectorStoreIndex
#
# index = VectorStoreIndex.from_documents(
#     documents, transformations=[text_splitter, title_extractor, qa_extractor],
# )
# print(index.as_retriever().retrieve("萧薰儿的斗气是多少？"))