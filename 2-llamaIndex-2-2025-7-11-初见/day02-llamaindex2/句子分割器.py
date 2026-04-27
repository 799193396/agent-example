from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 初始化分割器
splitter = SentenceSplitter(
    chunk_size=100,  # 分割长度
    chunk_overlap=20,  # 重叠长度
    paragraph_separator="\n\n",  # 段落分割符
    separator="，"  # 句子分割符
)
# 读取文件
documents = SimpleDirectoryReader(input_files=['data/小说.txt']).load_data()

# 分割文段
nodes = splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(node.text, "---"*10)