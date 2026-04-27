from llama_index.readers.json import JSONReader
from llama_index.core.node_parser import JSONNodeParser, SentenceSplitter
# 如果想保留原有的json格式，需要设置clean_json=False
reader = JSONReader(clean_json=False)

documents = reader.load_data(input_file="data/request.json")
print(documents)
# JSONNodeParser是专门解析json文档
# 如果想使用JSONNodeParser，需要设置 JSONReader(clean_json=False)
print(JSONNodeParser().get_nodes_from_documents(documents))
# s = SentenceSplitter(chunk_size=10, chunk_overlap=5)
# print(s.get_nodes_from_documents(documents))