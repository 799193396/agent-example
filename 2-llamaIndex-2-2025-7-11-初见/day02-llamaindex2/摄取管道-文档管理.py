from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import SentenceSplitter

# 读取指定文档
documents = SimpleDirectoryReader("data1", filename_as_id=True).load_data()
# 创建摄取管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5"),
    ],
    # 添加文档管理
    docstore=SimpleDocumentStore(),
)
# 执行管道
nodes = pipeline.run(documents=documents)

print(f"Ingested {len(nodes)} Nodes")

# 存储本地缓存
pipeline.persist("./pipeline_storage")
# 在加载文件之前，创建一个新的文件
with open('data1/t4.txt', 'w', encoding='utf-8') as f:
    f.write("这是测试文件3")

# 加载文件
documents1 = SimpleDirectoryReader("data1", filename_as_id=True).load_data()

# 创建新的摄取管道
pipeline1 = IngestionPipeline(
    transformations=[
        SentenceSplitter(),  # 句子拆分器
        HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5"),  # Hugging Face嵌入模型
    ])
# 恢复管道
pipeline1.load("./pipeline_storage")

nodes = pipeline1.run(documents=documents1)

print(f"Ingested {len(nodes)} Nodes")