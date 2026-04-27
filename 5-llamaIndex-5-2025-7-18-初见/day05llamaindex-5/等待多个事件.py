from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.workflow import (
    Context,
    Event,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.response_synthesizers import get_response_synthesizer
from typing import List
import asyncio
from util import load_model


# 定义工作流中的事件类型
class QueryEvent(Event):
    """查询事件"""
    query: str


class VectorRetrievalEvent(Event):
    """向量检索事件"""
    nodes: List[NodeWithScore]
    query: str


class BM25RetrievalEvent(Event):
    """关键词检索事件"""
    nodes: List[NodeWithScore]
    query: str


class PostProcessEvent(Event):
    """后处理事件"""
    processed_nodes: List[NodeWithScore]
    query: str


class ResponseEvent(Event):
    """响应事件"""
    response: str
    source_nodes: List[NodeWithScore]


class RAGWorkflow(Workflow):
    """RAG工作流类"""

    def __init__(self, retriever: VectorIndexRetriever, bm25_retriever: BM25Retriever):
        super().__init__()
        self.retriever = retriever
        self.bm25_retriever = bm25_retriever
        self.postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)
        self.response_synthesizer = get_response_synthesizer()

    @step
    async def query_step(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """
        步骤1: 处理用户查询
        """
        query = ev.query
        print(f"🔍 接收查询: {query}")

        processed_query = query.strip()

        return QueryEvent(query=processed_query)

    @step
    async def vector_retrieval_step(self, ctx: Context, ev: QueryEvent) -> VectorRetrievalEvent | None:
        """
        步骤2: vector向量数据库检索相关文档
        """
        print(f"📚 开始检索相关文档...")

        # 创建查询束
        query_bundle = QueryBundle(query_str=ev.query)

        # 执行检索
        retrieved_nodes = await self.retriever.aretrieve(query_bundle)

        return VectorRetrievalEvent(nodes=retrieved_nodes, query=ev.query)

    @step
    async def bm25_retrieval_step(self, ctx: Context, ev: QueryEvent) -> BM25RetrievalEvent | None:
        """
        步骤2: bm25检索相关文档
        """
        print(f"📚 bm25开始检索相关文档...")

        # 创建查询束
        query_bundle = QueryBundle(query_str=ev.query)

        # 执行检索
        retrieved_nodes = await self.retriever.aretrieve(query_bundle)

        print(f"✅ 检索到 {len(retrieved_nodes)} 个相关文档片段")

        return BM25RetrievalEvent(nodes=retrieved_nodes, query=ev.query)

    @step
    async def postprocess_step(self, ctx: Context, ev: VectorRetrievalEvent | BM25RetrievalEvent) -> PostProcessEvent:
        """
        步骤3: 对检索结果进行后处理
        """
        print(f"🔧 开始后处理检索结果...")
        # 等待Vector检索事件
        vector_events = ctx.collect_events(ev, [VectorRetrievalEvent])
        print(f"✅ 已收集Vector检索事件")

        # 等待BM25检索事件
        bm25_events = ctx.collect_events(ev, [BM25RetrievalEvent])
        print(f"✅ 已收集BM25检索事件")

        # 合并所有检索结果
        all_nodes = []
        query = ""

        if vector_events:
            all_nodes.extend(vector_events[0].nodes)
            query = vector_events[0].query
            print(f"  - Vector检索: {len(vector_events[0].nodes)} 个节点")

        if bm25_events:
            all_nodes.extend(bm25_events[0].nodes)
            query = bm25_events[0].query
            print(f"  - BM25检索: {len(bm25_events[0].nodes)} 个节点")

        if not all_nodes:
            print("⚠️  没有找到任何检索结果")
            return PostProcessEvent(processed_nodes=[], query=query)

        print(f"🔄 开始后处理 {len(all_nodes)} 个文档片段...")

        # 创建查询束用于后处理
        query_bundle = QueryBundle(query_str=query)

        # 执行后处理（去重、过滤、重排序等）
        processed_nodes = self.postprocessor.postprocess_nodes(
            nodes=all_nodes, query_bundle=query_bundle
        )

        print(f"✅ 后处理完成，保留 {len(processed_nodes)} 个高质量文档片段")

        # 打印每个节点的相似度分数
        for i, node in enumerate(processed_nodes[:3]):  # 只显示前3个
            score = node.score if node.score else 0
            print(f"  - 文档片段 {i + 1}: 相似度 {score:.3f}")

        return PostProcessEvent(processed_nodes=processed_nodes, query=query)

    @step
    async def synthesis_step(self, ctx: Context, ev: PostProcessEvent) -> StopEvent:
        """
        步骤4: 基于检索到的上下文生成最终答案
        """
        print(f"🤖 开始生成答案...")

        if not ev.processed_nodes:
            return StopEvent(result={
                "response": "抱歉，没有找到相关信息来回答您的问题。",
                "source_nodes": []
            })

        # 创建查询束
        query_bundle = QueryBundle(query_str=ev.query)

        # 使用响应合成器生成答案
        response = await self.response_synthesizer.asynthesize(
            query=query_bundle,
            nodes=ev.processed_nodes
        )

        print(f"✅ 答案生成完成")

        return StopEvent(result={
            "response": str(response),
            "source_nodes": ev.processed_nodes,
            "metadata": {
                "num_sources": len(ev.processed_nodes),
                "query": ev.query
            }
        })


# 使用示例
async def main():
    """主函数示例"""

    # 1. 准备数据和索引（这里使用示例数据）
    print("📖 正在构建向量索引...")

    # 加载大模型和嵌入模型
    load_model.get_llm()

    # 加载文件
    documents = SimpleDirectoryReader(input_files=["data/小说.txt"]).load_data()
    # 初始化节点解析器
    splitter = SentenceSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)

    # 创建向量索引对象
    index = VectorStoreIndex.from_documents(documents)
    # 创建向量检索器
    retriever = VectorIndexRetriever(index, similarity_top_k=5)
    # 创建关键字BM25检索器
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=3
    )

    print("✅ 向量索引构建完成")

    # 2. 创建并运行工作流
    workflow = RAGWorkflow(retriever=retriever, bm25_retriever=bm25_retriever)

    # 测试查询
    test_queries = [
        "萧炎的爸爸是谁？",
        "萧炎的妹妹是谁？"
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"🎯 测试查询: {query}")
        print(f"{'=' * 50}")

        # 运行工作流
        result = await workflow.run(query=query)

        # 显示结果
        print(f"\n📝 生成的答案:")
        print(f"{result['response']}")
        print(f"\n📊 元数据:")
        print(f"- 使用了 {result['metadata']['num_sources']} 个文档片段")
        print(f"- 原始查询: {result['metadata']['query']}")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())