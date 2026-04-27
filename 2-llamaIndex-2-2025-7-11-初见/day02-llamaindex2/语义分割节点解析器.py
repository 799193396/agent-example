from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


documents = SimpleDirectoryReader(input_files=['data/小说.txt']).load_data()
# 不会改变原有语句的意思，会根据Embedding去进行相似性判断。
embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

nodes = splitter.get_nodes_from_documents(documents)
# 打印生成的节点
for node in nodes:
    print(node.text, node.metadata, "------")