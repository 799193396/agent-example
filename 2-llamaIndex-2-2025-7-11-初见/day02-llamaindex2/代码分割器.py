from llama_index.core.node_parser import CodeSplitter
from llama_index.core import SimpleDirectoryReader

# 读取文件
documents = SimpleDirectoryReader(input_files=['data/demo.py']).load_data()
# 初始化代码分割器
splitter = CodeSplitter(
    language="python",
    chunk_lines=50,  # 每块行数
    chunk_lines_overlap=10,  # 重叠的数量
    max_chars=100,  # 块最大的数量
)
"""
    优先保证代码的完整性，如果超出所设的值，那就会强制切分
"""
# 将文档转换成节点
nodes = splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(f"Type: {node.metadata}\nText: {node.text}\n{'='*50}")