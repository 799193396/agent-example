from llama_index.core import SimpleDirectoryReader
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core import StorageContext
from util import load_model

# 加载大模型和嵌入模型
llm, embed_model = load_model.get_llm()

# 加载文档
documents = SimpleDirectoryReader(input_files=["data/小说.txt"]).load_data()
# 创建简单文档存储，并把节点传入
print(documents)
# 初始化redis文档存储对象  namespace=命名空间
doc_store = RedisDocumentStore.from_host_and_port(
    host="127.0.0.1", port=6379, namespace="llama_index"
)
# 添加文档到redis中
doc_store.add_documents(documents)
print(f"已存储文档: {doc_store.docs}")

# 创建存储的上下文
storage_context = StorageContext.from_defaults(docstore=doc_store)

print(len(storage_context.docstore.docs))

print("----------------------直接查询 Redis 数据库----------------------")
import redis

redis_client = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# 查看所有 keys
all_keys = redis_client.keys("*llama_index*")
print(f"Redis 中的所有相关 keys: {len(all_keys)} 个")

# 查看前几个 key 的内容
for key in all_keys[:3]:
    value = redis_client.hgetall(key)
    print(f"Key: {key}")
    print(f"Value: {value}...")
    print("-" * 30)