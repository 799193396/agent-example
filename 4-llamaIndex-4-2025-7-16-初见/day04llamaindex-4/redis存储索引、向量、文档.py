from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import VectorStoreIndex
from redisvl.schema import IndexSchema
from util import load_model

# 加载大模型和嵌入模型
llm, embed_model = load_model.get_llm()

# 设置向量存储的规则
custom_schema = IndexSchema.from_dict(
    {
        "index": {"name": "redis_vector_store", "prefix": "doc"},
        # 自定义被索引的字段
        "fields": [
            # llamaIndex的必填字段
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 1024,  # 向量维度， 这个维度需要和嵌入模型的维度一致，如果不一致会出现错误
                    "algorithm": "hnsw",  # 算法
                    "distance_metric": "cosine",  # 相似度计算：余弦
                },
            },
        ],
    }
)


def create_and_store_index():
    """创建并存储索引的完整流程"""

    # 重新加载文档（确保数据新鲜）
    documents = SimpleDirectoryReader(input_files=["data/小说.txt"]).load_data()
    nodes = SentenceSplitter().get_nodes_from_documents(documents)

    # 创建存储组件
    storage_context = StorageContext.from_defaults(
        index_store=RedisIndexStore.from_host_and_port(
            host="127.0.0.1", port=6379, namespace="novel_index"
        ),
        docstore=RedisDocumentStore.from_host_and_port(
            host="127.0.0.1", port=6379, namespace="novel_docs"
        ),
        vector_store=RedisVectorStore(
            schema=custom_schema,
            redis_url="redis://127.0.0.1:6379"
        )
    )

    # 创建索引，当进行创建索引的时候，会自动将文档、索引、向量存到redis中
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    print(f"✅ 索引创建并存储完成，ID: {index.index_id}")
    # 测试查询
    response = index.as_query_engine().query("萧炎的戒指是谁给他的")
    print(f"✅ 加载成功！查询结果: {response}")

    return index.index_id


def load_and_query_index(index_id=None):
    """加载并查询索引"""
    """
        一般做企业中的RAG，索引会只有一个
    """
    # 创建相同配置的存储上下文，加载对应的数据，命名空间一定要一致
    storage_context = StorageContext.from_defaults(
        index_store=RedisIndexStore.from_host_and_port(
            host="127.0.0.1", port=6379, namespace="novel_index"
        ),
        docstore=RedisDocumentStore.from_host_and_port(
            host="127.0.0.1", port=6379, namespace="novel_docs"
        ),
        vector_store=RedisVectorStore(
            schema=custom_schema,
            redis_url="redis://localhost:6379"
        )
    )

    try:
        # 加载索引
        if index_id:
            loaded_index = load_index_from_storage(storage_context, index_id=index_id)
        else:
            # 多个索引对应会报错
            loaded_index = load_index_from_storage(storage_context)

        # 测试查询
        response = loaded_index.as_query_engine().query("是谁要被退婚？")
        print(f"✅ 加载成功！查询结果: {response}")

        return loaded_index

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


# 1. 创建和存储
stored_index_id = create_and_store_index()

# 2. 加载和查询
loaded_index = load_and_query_index(stored_index_id)

if loaded_index:
    print("🎉 完整流程成功！")
else:
    print("❌ 流程失败")
