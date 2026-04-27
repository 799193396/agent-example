from typing import List

from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from util import load_model

llm, embed_model = load_model.get_llm()

# 加载文档并构建索引
documents = SimpleDirectoryReader(
    input_files=["data/小说.txt"]
).load_data()

# 创建提取规则
kg_extractor = SimpleLLMPathExtractor(
     llm=llm,
     max_paths_per_chunk=10,  # 控制从每个文档块(chunk)中最多提取多少条路径
     num_workers=4,  # 并行数量
)

print("kg_extractor->", kg_extractor)

# 创建属性图
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractor=kg_extractor,
    show_progress=True  # 显示提取进度
)
# 查看结果
response = index.property_graph_store.get_triplets(entity_names=["萧炎"])
print("response->", response)