from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event
)
from pathlib import Path

from llama_index.core.schema import NodeWithScore
from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

load_dotenv()
# 配置全局设置
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")
model = "qwen-plus-2025-01-25"
Settings.llm = DashScope(model=model, api_key=api_key, api_base=api_base_url, is_chat_model=True, timeout=60)
Settings.embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")


class QueryEvent(Event):
    """查询事件"""
    query: str


class RetrievalEvent(Event):
    """检索事件"""
    nodes: List[NodeWithScore]
    query: str


class RAGWorkflow(Workflow):
    """RAG 工作流"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _check_existing_index(self, file_path: str) -> bool:
        """检查是否已存在索引"""
        persist_path = Path("./storage")
        # 判断是否存在当前索引对象，如果不存在则返回false
        if not persist_path.exists():
            return False

        # 检查索引文件是否存在
        required_files = ["docstore.json", "index_store.json", "default_vector_store.json"]
        if not all((persist_path / file).exists() for file in required_files):
            return False

        # 可选：检查文件修改时间，如果源文件更新则重建索引
        try:
            source_mtime = os.path.getmtime(file_path)
            index_mtime = os.path.getmtime(persist_path / "docstore.json")
            if source_mtime > index_mtime:
                print("源文件已更新，需要重建索引")
                return False
        except OSError:
            return False

        return True

    def _load_existing_index(self):
        """加载已存在的索引"""
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        return load_index_from_storage(storage_context)

    @step
    async def load_documents(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """步骤1: 加载文档并构建索引"""
        file_path = ev.get("file_path", "D:\llm\LLMProject\LlamaIndex\data\小说.txt")
        query = ev.get("query", "")
        print(f"加载文档: {file_path}")

        # 1. 检查是否已存在索引（避免重复构建）
        if self._check_existing_index(file_path):
            print("发现已存在的索引，直接加载...")
            index = self._load_existing_index()
            await ctx.store.set("index", index)
            print("索引加载完成")
            return QueryEvent(query=query)

        # 加载文档
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        # 构建向量索引
        index = VectorStoreIndex.from_documents(documents)

        # 存储到上下文中
        await ctx.store.set("index", index)

        print(f"加载 {len(documents)} 文档到索引中")

        # 返回查询事件，这里可以根据需要设置默认查询
        return QueryEvent(query=query)

    @step
    async def retrieve_documents(self, ctx: Context, ev: QueryEvent) -> RetrievalEvent:
        """步骤2: 检索相关文档"""
        if not ev.query:
            return StopEvent(result="请输入问题")

        print(f"检索要查询的文档: {ev.query}")

        # 从上下文获取索引
        index = await ctx.store.get("index")
        if not index:
            return StopEvent(result="没有找到索引。请先载入文件.")

        # 创建检索器
        retriever = index.as_retriever(similarity_top_k=5)

        # 检索相关节点
        try:
            nodes = await retriever.aretrieve(ev.query)
        except Exception as e:
            return StopEvent(result=f"检索出错: {e}")

        print(f"检索到 {len(nodes)} 个文档")

        return RetrievalEvent(query=ev.query, nodes=nodes)

    @step
    async def generate_response(self, ctx: Context, ev: RetrievalEvent) -> StopEvent:
        """步骤3: 生成回答"""
        print(f"根据问题响应答案: {ev.query}")

        # 从上下文获取索引
        index = await ctx.store.get("index")

        # 创建查询引擎
        query_engine = index.as_query_engine()

        # 生成回答
        try:
            response = await query_engine.aquery(ev.query)
        except Exception as e:
            return StopEvent(result=f"生成回答出错: {e}")
        print(response)
        return StopEvent(result=str(response))


# `workflow` 会被LlamaDeploy导入
rag_workflow = RAGWorkflow()


async def main():
    print(await rag_workflow.run(file_path="D:\llm\LLMProject\LlamaIndex\data\小说.txt", query="萧炎的爸爸是谁？"))


# 让这个脚本可以在shell中运行，这样我们就可以测试工作流的执行了
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())