from llama_index.core import SummaryIndex, Document
from util import load_model

llm, embed_model = load_model.get_llm()


# 准备示例文档数据
documents = [
    Document(text="""
    人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的智能机器。
    AI 包括机器学习、深度学习、自然语言处理等多个子领域。
    机器学习是 AI 的核心技术之一，通过算法让计算机从数据中学习模式。
    """),

    Document(text="""
    深度学习是机器学习的一个子集，使用人工神经网络来模拟人脑的工作方式。
    深度学习在图像识别、语音识别和自然语言处理方面取得了突破性进展。
    卷积神经网络（CNN）特别适合处理图像数据，循环神经网络（RNN）适合处理序列数据。
    """),

    Document(text="""
    自然语言处理（NLP）是 AI 的一个重要分支，专注于让计算机理解和生成人类语言。
    NLP 的应用包括机器翻译、情感分析、文本摘要和问答系统。
    现代 NLP 系统大多基于 Transformer 架构，如 GPT 和 BERT 模型。
    """)
]
# 创建 SummaryIndex-摘要索引
summary_index = SummaryIndex.from_documents(documents)

# 4. 执行查询
print("\n=== 执行查询 ===")

# 查询 1: 总体概述
print("查询 1: 什么是人工智能？")
response1 = summary_index.as_query_engine().query("什么是人工智能，包括哪些主要技术？")
print(f"回答: {response1}")

print("\n" + "="*50 + "\n")

# 查询 2: 特定技术
print("查询 2: 深度学习的应用领域")
response2 = summary_index.as_query_engine().query("深度学习在哪些领域有应用？")
print(f"回答: {response2}")

print("\n" + "="*50 + "\n")

# 查询 3: 技术对比
print("查询 3: 不同 AI 技术的关系")
response3 = summary_index.as_query_engine().query("机器学习、深度学习和自然语言处理之间的关系是什么？")
print(f"回答: {response3}")

# 查看索引结构信息
print("\n=== 索引结构信息 ===")
print(f"文档数量: {len(summary_index.docstore.docs)}")
print(f"节点数量: {len(summary_index.index_struct.nodes)}")


print("\n=== 添加新文档 ===")
new_doc = Document(text="""
计算机视觉是人工智能的另一个重要分支，致力于让计算机能够识别和理解图像和视频。
计算机视觉的应用包括人脸识别、物体检测、图像分类和自动驾驶。
现代计算机视觉系统主要基于深度学习技术，特别是卷积神经网络。
""")

summary_index.insert(new_doc)
print("新文档已添加到索引")
print(f"更新后的文档数量: {len(summary_index.docstore.docs)}")

# 查询新添加的内容
print("\n查询新内容:")
response_new = summary_index.as_query_engine().query("计算机视觉有哪些应用？")
print(f"回答: {response_new}")