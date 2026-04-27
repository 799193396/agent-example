from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from util import load_model

# 加载大模型和嵌入模型
llm, embed_model = load_model.get_llm()

# 加载文档
documents = SimpleDirectoryReader(input_files=["data/小说.txt"]).load_data()
# 初始化节点解析器
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)
# 创建文档存储器
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

# 创建chroma连接对象
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 创建上下文存储器
storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)
# 创建向量索引
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
# 创建混合检索器
retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=2),
        BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=2
        ),
    ],
    mode=FUSION_MODES.RECIPROCAL_RANK,
    # 根据问题生成的问题数量，设置为1就是禁用
    num_queries=4,
    similarity_top_k=2,
    use_async=True,
)

nodes_with_scores = retriever.retrieve("纳兰嫣然在哪个宗门修炼？")
for node in nodes_with_scores:
    print(f"Score: {node.score:.2f} - {node.text}...\n-----\n")

# 创建检索查询引擎
query_engine = RetrieverQueryEngine(retriever)
print(query_engine.query("纳兰嫣然在哪个宗门修炼？"))