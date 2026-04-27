from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.json import JSONReader

# 定义数据连接器去读取数据
reader = JSONReader()
documents = reader.load_data(input_file="data/request.json")
# 定义本地化的向量化
# 加载本地的嵌入模型
embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")
# 定义文本分割器
text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

# 创建数据摄入管道
pipeline = IngestionPipeline(
    transformations=[text_splitter, embed_model]
)

# 执行管道
nodes = pipeline.run(documents=documents)

# 打印处理后的节点
for node in nodes:
    print(node, "-------", "\n\n")