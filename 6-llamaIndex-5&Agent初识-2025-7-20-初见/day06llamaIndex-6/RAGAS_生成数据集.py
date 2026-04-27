from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from datasets import Dataset
import pandas as pd


def generate_question_answer_pairs(llm, text):
    """大模型根据对应文档内容生成问题和答案的工具"""
    prompt = f"""
                你是一个文档理解助手。请根据以下内容，提出一个清晰的问题，并给出准确的答案。

                内容：
                {text}

                请按以下格式输出：
                Question: ...
                Answer: ...
            """
    response = llm.complete(prompt).text
    try:
        # 将大模型生成的数据库进行分割
        lines = response.strip().split("\n")
        question_line = None
        answer_line = None

        # 通过字符串的startswith（根据以什么开头的）获取对应的问题还是答案
        for line in lines:
            if line.startswith("Question:"):
                question_line = line
            elif line.startswith("Answer:"):
                answer_line = line

        if question_line and answer_line:
            question = question_line.replace("Question:", "").strip()
            answer = answer_line.replace("Answer:", "").strip()
            return question, answer
        else:
            return None, None
    except Exception as e:
        print(f"解析问答对时出错: {e}")
        return None, None


def get_ragas_datas(llm, documents):
    """将大模型生成的数据转变成RAGAS支持的数据集"""
    # 切分为句子粒度（用于提问）
    parser = SentenceSplitter(chunk_size=200, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)

    # 构建索引
    index = VectorStoreIndex(nodes)

    # 获取检索器
    retriever = index.as_retriever(similarity_top_k=3)
    query_engine = index.as_query_engine()

    # 构造 RAGAS 数据格式
    ragas_records = []
    successful_count = 0
    failed_count = 0

    for i, node in enumerate(nodes[:3]):  # 可选：限制数量调试
        try:
            context = node.text.strip()

            # 跳过太短的文本
            if len(context) < 50:
                continue

            question, ground_truth = generate_question_answer_pairs(llm, context)

            if not question or not ground_truth:
                failed_count += 1
                print(f"节点 {i} 生成问答对失败")
                continue

            # 检索 top-k 作为 retrieved_context
            retrieved_nodes = retriever.retrieve(question)
            # 检索出对应的上下文
            retrieved_contexts = [n.text.strip() for n in retrieved_nodes]

            # 确保数据格式正确
            record = {
                "question": str(question).strip(),  # 问题
                "answer": str(query_engine.query(question)).strip(),  # RAG系统生成的回复
                "contexts": retrieved_contexts,  # 保持为列表格式，检索的上下文
                "ground_truth": str(ground_truth).strip()  # 正确答案
            }

            # 验证记录的完整性
            if all(record[key] for key in ["question", "ground_truth", "answer", "contexts"]) and record["contexts"]:
                ragas_records.append(record)
                successful_count += 1
            else:
                failed_count += 1
                print(f"节点 {i} 数据不完整，跳过")

        except Exception as e:
            failed_count += 1
            print(f"处理节点 {i} 时出错: {e}")
            continue

    print(f"成功生成 {successful_count} 条记录，失败 {failed_count} 条")

    if not ragas_records:
        print("警告：没有生成任何有效记录")
        return None

    # 输出为 DataFrame
    df_ragas = pd.DataFrame(ragas_records)
    eval_dataset = Dataset.from_pandas(df_ragas)

    # 可选：保存到文件
    # df_ragas.to_json("ragas_dataset_llm.json", orient="records", force_ascii=False, indent=2)

    print("成功生成 RAGAS 格式数据（带 LLM 问答对）")
    return eval_dataset

"""
    llamaindex对RAGAS评估支持不是很好，借助langchain中的加载模型的模块去辅助
"""