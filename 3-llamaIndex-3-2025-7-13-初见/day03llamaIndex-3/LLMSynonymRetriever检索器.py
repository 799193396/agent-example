from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.indices.property_graph import LLMSynonymRetriever
from llama_index.core import PropertyGraphIndex
from llama_index.core import Document
from typing import Literal
from util import load_model

llm, embed_model = load_model.get_llm()
# 定义提取模式
doc = [
    Document(text="张伟发表了多篇关于深度学习的论文，他的研究团队包括3名博士生和5名硕士生。"),
    Document(text="李娜在阿里巴巴的项目涉及多模态AI，她和王强共同负责模型优化部分。"),
    Document(text="北京大学人工智能学院与阿里巴巴达摩院建立了合作关系，共同推进AI技术发展。"),
    Document(text="王强之前在腾讯工作，后来跳槽到阿里巴巴，专注于大模型推理加速技术。"),
    Document(text="张伟的研究领域还包括计算机视觉和强化学习，他指导的学生分布在各大科技公司。")
]

# 定义提取模式
entities = Literal[
    "NAME",  # 人员
    "ORGANIZATION",  # 组织机构
    "POSITION",  # 职位
    "RESEARCH_FIELD",  # 研究领域
    "PROJECT",  # 项目
    "LOCATION",  # 地点
    "TECHNOLOGY",  # 技术
    "PUBLICATION"  # 出版物
]
relations = Literal[
    "WORKS_AT",  # 工作于
    "STUDIES_AT",  # 学习于
    "RESEARCHES",  # 研究
    "SUPERVISES",  # 指导
    "COLLABORATES_WITH",  # 合作
    "PARTICIPATES_IN",  # 参与
    "LOCATED_IN",  # 位于
    "SPECIALIZES_IN",  # 专业于
    "PUBLISHED",  # 发表
    "DEVELOPS",  # 开发
    "LEADS",  # 领导
    "MEMBER_OF",  # 成员
    "PARTNER_WITH"  # 合作伙伴
]
# 定义更详细的图谱模式
schema = {
    "NAME": [
        "WORKS_AT", "STUDIES_AT", "RESEARCHES", "SUPERVISES",
        "COLLABORATES_WITH", "PARTICIPATES_IN", "SPECIALIZES_IN",
        "PUBLISHED", "DEVELOPS", "LEADS"
    ],
    "ORGANIZATION": [
        "LOCATED_IN", "SPECIALIZES_IN", "COLLABORATES_WITH",
        "DEVELOPS", "PARTNER_WITH"
    ],
    "POSITION": ["LOCATED_IN", "SPECIALIZES_IN"],
    "RESEARCH_FIELD": ["DEVELOPS", "SPECIALIZES_IN"],
    "PROJECT": ["DEVELOPS", "COLLABORATES_WITH"],
    "LOCATION": ["LOCATED_IN"],
    "TECHNOLOGY": ["DEVELOPS", "SPECIALIZES_IN"],
    "PUBLICATION": ["RESEARCHES", "PUBLISHED"]
}
# 创建基于模式的提取器
kg_extractor = SchemaLLMPathExtractor(llm=llm,
                                      possible_entities=entities,
                                      possible_relations=relations,
                                      kg_validation_schema=schema,
                                      strict=False,  # 如果为 false，将允许超出模式范围的三元组
                                      num_workers=4,  # 并行处理
                                      )

# 创建属性图
index = PropertyGraphIndex.from_documents(
    doc,
    kg_extractor=kg_extractor,
    show_progress=True  # 显示提取进度
)
synonym_retriever = LLMSynonymRetriever(
    index.property_graph_store,
    llm=llm,
    # 包括检索路径的源块文本
    include_text=True,
    max_keywords=10,  # 要生成的同义词的数量
    # 节点检索后要遵循的关系深度
    path_depth=2,
)

retriever = index.as_retriever(sub_retrievers=[synonym_retriever])
print(retriever.retrieve("张伟？"))