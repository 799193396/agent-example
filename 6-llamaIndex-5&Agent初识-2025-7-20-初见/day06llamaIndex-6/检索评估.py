from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import asyncio
import pandas as pd
from util import load_model

# 加载大模型和嵌入模型
llm, embed_model = load_model.get_llm()


async def main():
    def display_results(name, eval_results):
        """显示evaluate的结果"""

        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        columns = {
            "retrievers": [name],
            **{k: [full_df[k].mean()] for k in metrics},
        }

        metric_df = pd.DataFrame(columns)

        return metric_df

    # 加载文档
    documents = SimpleDirectoryReader(input_files=["data/小说.txt"]).load_data()

    # 将文档解析为节点
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)

    # 传递嵌入模型
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=2)

    qa_generate_prompt_tmpl = """
        上下文信息如下：
        ---------------------
        {context_str}
        ---------------------
        根据上述上下文信息，而非先验知识，仅基于以下查询生成问题。
        您是一位教师/教授。您的任务是为即将到来的测验/考试准备{num_questions_per_chunk}道题目。题目应该在整个文档范围内具有多样化的性质。请将问题限制在所提供的上下文信息范围内。
    """

    print("正在生成评估数据集...")
    qa_dataset = generate_question_context_pairs(
        nodes, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl, llm=llm, num_questions_per_chunk=2
    )

    # 保存数据集
    qa_dataset.save_json("pg_eval_dataset.json")

    # 平均倒数排名=mrr（最有用的文档排在第几位）， 命中率=hit_rate（检索的文档里，有没有包含正确答案的）， 精确度（检索回来的文档中，有几个是真正有用的）=precision
    metrics = ["hit_rate", "mrr", "precision"]
    # 创建检索器评估
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metrics, retriever=retriever
    )
    # 开始评估
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    # 显示结果
    print(display_results("top-1 eval", eval_results))


if __name__ == '__main__':
    asyncio.run(main())