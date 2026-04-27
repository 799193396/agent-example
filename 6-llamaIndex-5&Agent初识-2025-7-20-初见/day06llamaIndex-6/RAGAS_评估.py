from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from RAGAS_生成数据集 import get_ragas_datas
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader, Settings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

load_dotenv()

# 加载大模型和嵌入模型
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")
"""
LlamaIndex + RAGAS 评估案例
这个例子展示了如何使用LlamaIndex构建RAG系统，并用RAGAS进行评估
"""


class RAGEvaluator:
    def __init__(self):
        # 设置LlamaIndex配置 - 使用Qwen模型
        self.llm = DashScope(model_name="qwen-plus-1127", api_key=api_key, api_base=api_base_url)
        Settings.llm = self.llm

        # 加载本地的嵌入模型
        self.embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")
        # 设置默认的向量模型为本地模型
        Settings.embed_model = self.embed_model

        # 创建RAGAS适配器
        self.ragas_llm = ChatOpenAI(
            model="qwen-plus-1127",
            api_key=api_key,
            base_url=api_base_url,
            temperature=0.1
        )
        # RAGAS需要langchain格式的嵌入模型
        self.ragas_embeddings = HuggingFaceEmbeddings(
            model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5",
            model_kwargs={'trust_remote_code': True}
        )

    def evaluate_with_ragas(self, eval_dataset):
        """使用RAGAS评估RAG系统"""
        print("开始RAGAS评估...")

        # 定义评估指标
        metrics = [
            answer_relevancy,  # 答案相关性
            faithfulness,  # 忠实度
            context_recall,  # 上下文召回率
            context_precision,  # 上下文精确度
        ]

        # 使用RAGAS评估 - 需要指定LLM和Embeddings
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
        )

        return result


def main():
    """主函数"""
    try:
        # 初始化评估器
        evaluator = RAGEvaluator()

        # 加载文档和构建索引
        documents = SimpleDirectoryReader(input_files=["data/小说.txt"]).load_data()

        eval_dataset = get_ragas_datas(llm=Settings.llm, documents=documents)
        # 显示生成的回答
        print("\n=== 生成的数据 ===")
        print(eval_dataset.to_pandas().head())

        # 使用RAGAS评估
        result = evaluator.evaluate_with_ragas(
            eval_dataset
        )
        # result.upload()
        results_df = result.to_pandas()
        results_df.to_csv("ragas_evaluation_results.csv", index=False)
        print("\n详细结果已保存到 ragas_evaluation_results.csv")
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()