from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from util import load_model

# 加载大模型和嵌入模型
llm, embed_model = load_model.get_llm()

# 加载文档
# 加载文档和构建索引
documents = SimpleDirectoryReader(input_files=["data/小说.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents)

# 使用同一个查询引擎（推荐）
query = "萧炎的爸爸是谁？"

# 创建查询引擎
query_engine = index.as_query_engine()

# 执行查询
response = query_engine.query(query)

# 从响应对象中获取源节点（这些是实际用于生成回答的上下文）
# if hasattr(response, 'source_nodes') and response.source_nodes:
#     # 使用实际参与回答生成的上下文
#     contexts = [node.text for node in response.source_nodes]
# else:
# 如果响应对象没有源节点，则手动检索
retriever = index.as_retriever()
retrieved_nodes = retriever.retrieve(query)
contexts = [node.text for node in retrieved_nodes]

# 输出对应的内容
print("上下文内容：", contexts)

print("生成的回复：", response)


# 初始化评估器
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

print("\n--- 开始评估忠实度 (Relevancy) ---")
# 生成答案对检索内容的忠实度（生成的答案是否是从检索出来的上下文生成的）
faithfulness_result = faithfulness_evaluator.evaluate(
    query=query,
    response=str(response),
    contexts=contexts  # 传入字符串列表
)

print(f"评估结果: {'通过' if faithfulness_result.passing else '未通过'}")
print(f"分数: {faithfulness_result.score}")
print(f"反馈: {faithfulness_result.feedback}")

print("\n--- 开始评估相关性 (Relevancy) ---")
# 生成的答案和原始问题的相关系
relevancy_result = relevancy_evaluator.evaluate(
    query=query,
    response=str(response),
    contexts=contexts
)

print(f"评估结果: {'通过' if relevancy_result.passing else '未通过'}")
print(f"分数: {relevancy_result.score}")
print(f"反馈: {relevancy_result.feedback}")

# --- 4. 可选：批量评估 ---
print("\n--- 批量评估示例 ---")
queries = ["萧炎的爸爸是谁？", "萧炎的实力如何？"]

for q in queries:
    print(f"\n处理查询: {q}")
    # 创建查询引擎
    query_engine = index.as_query_engine()
    # 执行查询
    resp = query_engine.query(q)

    # 从响应对象中获取源节点（这些是实际用于生成回答的上下文）
    if hasattr(resp, 'source_nodes') and resp.source_nodes:
        # 使用实际参与回答生成的上下文
        ctxs = [node.text for node in resp.source_nodes]
    else:
        # 如果响应对象没有源节点，则手动检索
        retriever = index.as_retriever()
        retrieved_nodes = retriever.retrieve(query)
        ctxs = [node.text for node in retrieved_nodes]

    # 快速评估
    faith_result = faithfulness_evaluator.evaluate(
        query=q, response=str(resp), contexts=ctxs
    )
    rel_result = relevancy_evaluator.evaluate(
        query=q, response=str(resp), contexts=ctxs
    )

    print(f"忠实度: {faith_result.score:.2f}, 相关性: {rel_result.score:.2f}")


"""
    评估一定是基于多个问题，会囊括很多形式的问题去对RAG系统进行全方位的评估
"""