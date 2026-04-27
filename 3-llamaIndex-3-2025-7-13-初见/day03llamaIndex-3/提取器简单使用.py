from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from util import load_model

llm, embed_model = load_model.get_llm()

# 加载文档并构建索引
documents = SimpleDirectoryReader(
    input_files=["data/小说.txt"]
).load_data()

# 创建属性图
index = PropertyGraphIndex.from_documents(
    documents,
)

# 使用
retriever = index.as_retriever(
    include_text=True,  # 包括与匹配路径的源块
    similarity_top_k=2,  # 向量 kg 节点检索的前 k 个
)
nodes = retriever.retrieve("萧炎的爸爸是谁？")
print(nodes)
query_engine = index.as_query_engine(
    include_text=False,  # 包括与匹配路径的源块
    similarity_top_k=3,  # 向量 kg 节点检索的前 k 个
)
response = query_engine.query("萧炎的爸爸是谁？")
print("-" * 20)
print(response)