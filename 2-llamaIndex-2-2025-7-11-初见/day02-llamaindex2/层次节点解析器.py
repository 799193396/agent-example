from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core import SimpleDirectoryReader

# 读取数据
documents = SimpleDirectoryReader(input_files=['data/小说.txt']).load_data()

# 进行层次节点解析器 chunk_sizes=每层目标Token数（从粗到细）
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128, 50]
)
# 文档转换成节点
nodes = node_parser.get_nodes_from_documents(documents)
for node in nodes:
    print(f"ID: {node.node_id}, Text: {node.text}...")
    if node.parent_node:
        print(f"Parent: {node.parent_node.node_id}")