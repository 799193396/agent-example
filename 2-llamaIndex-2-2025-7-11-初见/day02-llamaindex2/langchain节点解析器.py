from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core import SimpleDirectoryReader

# 读取文件
documents = SimpleDirectoryReader(input_files=['data/小说.txt']).load_data()

# 包装LangChain中的递归切割文本
parser = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50))
nodes = parser.get_nodes_from_documents(documents)
print(nodes)