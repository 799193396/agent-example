from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re
import os

load_dotenv()


class ThinkingReflectionFramework:
    def __init__(self, model="qwen-max-0428", temperature=0.7):
        self.llm = ChatOpenAI(model=model,
                              api_key=os.getenv("DASHSCOPE_API_KEY"),
                              base_url=os.getenv("DASHSCOPE_BASE_URL"),
                              temperature=temperature)
        self.parser = StrOutputParser()

    def create_thinking_prompt(self):
        """创建思考阶段的提示词"""
        thinking_template = """
                    你需要解决以下问题，请按照以下步骤进行思考：

                    <thinking>
                    1. 问题分析：仔细分析问题的核心要求
                    2. 知识回顾：回顾相关的知识和概念
                    3. 解决方案：提出可能的解决方案
                    4. 步骤规划：制定详细的解决步骤
                    </thinking>

                    问题：{question}

                    请在<thinking>标签内详细展示你的思考过程，然后给出最终答案。
                """
        return ChatPromptTemplate.from_template(thinking_template)

    def create_reflection_prompt(self):
        """创建反思阶段的提示词"""
        reflection_template = """
                原问题：{question}

                初始答案：{initial_answer}

                请对上述答案进行反思和评估：

                <reflection>
                1. 答案准确性：分析答案是否准确回答了问题
                2. 逻辑完整性：检查推理逻辑是否完整
                3. 遗漏分析：是否遗漏了重要信息
                4. 改进建议：提出具体的改进建议
                </reflection>

                基于反思，请提供一个改进后的最终答案。
            """
        return ChatPromptTemplate.from_template(reflection_template)

    def extract_thinking_content(self, response):
        """提取思考内容"""
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        if thinking_match:
            return thinking_match.group(1).strip()
        return ""

    def extract_reflection_content(self, response):
        """提取反思内容"""
        reflection_match = re.search(r'<reflection>(.*?)</reflection>', response, re.DOTALL)
        if reflection_match:
            return reflection_match.group(1).strip()
        return ""

    def solve_with_thinking_reflection(self, question, max_iterations=2):
        """使用思考和反思框架解决问题"""
        print("=" * 60)
        print("🧠 思考和自我反思框架")
        print("=" * 60)

        # 第一阶段：思考
        print("\n🤔 思考中")
        print("-" * 30)

        thinking_prompt = self.create_thinking_prompt()
        thinking_chain = thinking_prompt | self.llm | self.parser

        initial_response = thinking_chain.invoke({"question": question})

        # 提取思考过程
        thinking_content = self.extract_thinking_content(initial_response)
        if thinking_content:
            print("思考过程:")
            print(thinking_content)

        print(f"\n初始答案:")
        print(initial_response.replace(f"<thinking>{thinking_content}</thinking>", "").strip())

        # 第二阶段：反思和改进
        current_answer = initial_response

        for i in range(max_iterations):
            print(f"\n🔄 反思中 {i + 1}")
            print("-" * 30)

            reflection_prompt = self.create_reflection_prompt()
            reflection_chain = reflection_prompt | self.llm | self.parser

            reflection_response = reflection_chain.invoke({
                "question": question,
                "initial_answer": current_answer
            })

            # 提取反思过程
            reflection_content = self.extract_reflection_content(reflection_response)
            if reflection_content:
                print("反思过程:")
                print(reflection_content)

            # 提取改进后的答案
            improved_answer = reflection_response.replace(f"<reflection>{reflection_content}</reflection>", "").strip()

            print(f"\n改进后答案:")
            print(improved_answer)

            current_answer = improved_answer

        return current_answer


# 使用示例
if __name__ == "__main__":
    framework = ThinkingReflectionFramework()

    # 测试问题1：数学逻辑问题
    question1 = "如果一个农夫有17只羊，除了9只以外都死了，那么还剩下多少只羊？"
    result1 = framework.solve_with_thinking_reflection(question1)

    print("\n" + "=" * 60)
    print("最终结果:", result1)

    # 测试问题2：复杂推理问题
    print("\n" + "=" * 80)
    question2 = "一个人每天工作8小时，一周工作5天。如果他的时薪是25元，那么他一个月（4周）能赚多少钱？但是如果他加班超过40小时，超出部分按1.5倍计算。假设他每周加班5小时，计算他的月收入。"
    result2 = framework.solve_with_thinking_reflection(question2)

    print("\n" + "=" * 60)
    print("最终结果:", result2)