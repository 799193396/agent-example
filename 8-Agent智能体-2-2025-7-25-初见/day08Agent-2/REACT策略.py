from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain_core.tools import tool
import numexpr
import math
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain
from dotenv import load_dotenv
import os

load_dotenv()


class ReActFramework:
    def __init__(self, model="qwen-max-0428", temperature=0.1):
        self.llm = ChatOpenAI(model=model,
                              api_key=os.getenv("DASHSCOPE_API_KEY"),
                              base_url=os.getenv("DASHSCOPE_BASE_URL"),
                              temperature=temperature)
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _create_tools(self):
        """创建ReAct代理使用的工具集"""

        # 1. 搜索工具
        search = SerpAPIWrapper()
        search_tool = Tool(
            name="Search",
            func=search.run,
            description="用于搜索当前信息、新闻、实时数据等。输入应该是搜索查询。"
        )

        # 2. Wikipedia工具
        wikipedia = WikipediaAPIWrapper(
            lang="zh",
            top_k_results=1,
            doc_content_chars_max=800  # 限制内容长度，避免token过多
        )
        wikipedia = WikipediaQueryRun(api_wrapper=wikipedia,
                                      name="维基百科搜索",
                                      description="用于搜索维基百科信息的工具，输入应该是搜索查询"
                                      )

        # 3. 计算工具
        # 创建一个数学计算工具
        @tool
        def calculator(expression: str) -> str:
            """支持多个表达式的数学计算器（使用 numexpr）"""
            try:
                local_dict = {"pi": math.pi, "e": math.e} # 获取计算中的常量
                expression_1, expression_2 = expression.split("\n", maxsplit=1)
                expressions = [expr.strip() for expr in expression_1.split(";") if expr.strip()]
                results = []

                for expr in expressions:
                    result = numexpr.evaluate(expr, global_dict={}, local_dict=local_dict)
                    results.append(f"{expr} = {result}")

                return "\n".join(results)
            except Exception as e:
                return f"计算错误: {str(e)}"

        # 4. 自定义知识工具
        def knowledge_base(query):
            """简单的知识库工具"""
            knowledge = {
                "小猫爱学工作时间": "每周日至周五下午一点至晚上十点半",
                "小猫爱学休假类型": "员工依法享有法定假，事假不计薪，病假：正式员工每年享有5天带薪病假",
                "小猫爱学工作纪律": "禁止穿奇装异服，不能私自离岗"
            }
            # 模糊匹配
            query_lower = query.lower()
            for key, value in knowledge.items():
                if key.lower() in query_lower:
                    return value

            return "抱歉，我的知识库中没有关于这个主题的信息。"

        knowledge_tool = Tool(
            name="Knowledge",
            func=knowledge_base,
            description="查询内部知识库，专门用于小猫爱学相关的问题。"
        )

        return [search_tool, wikipedia, calculator, knowledge_tool]

    def _create_agent(self):
        """创建ReAct代理"""
        # 获取ReAct提示模板
        # try:
        #     # prompt = hub.pull("hwchase17/react")
        # except:
        # 如果无法从hub获取，使用自定义模板
        from langchain_core.prompts import PromptTemplate

        template = """你是一个乐于助人的智能助手，在回答问题时应通过推理，并在需要时调用工具。

        你可以使用以下工具：

        {tools}

        请遵循以下格式（注意英文冒号不能省略）：

        Question: 你需要回答的输入问题  
        Thought: 请一步步思考如何解答这个问题  
        Action: 你要执行的操作，必须是以下之一 [{tool_names}]  
        Action Input: 传入该操作的输入  
        Observation: 操作执行后的返回结果  
        ...（Thought/Action/Action Input/Observation 的过程可以重复多次）  
        Thought: 我现在已经知道最终答案  
        Final Answer: 对最初问题的最终回答

        即使你觉得自己知道答案，也请优先考虑使用工具获取答案。

        现在开始！

        Question: {input}  
        {agent_scratchpad}
        """

        prompt = PromptTemplate.from_template(template)

        # 创建ReAct代理
        agent = create_react_agent(self.llm, self.tools, prompt)

        # 创建代理执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,  # 允许重试
            max_iterations=5,  # 减少迭代次数
            max_execution_time=45,  # 45秒超时
            early_stopping_method="generate",  # 早停机制
            return_intermediate_steps=True  # 返回中间步骤便于调试
        )

        return agent_executor


    def solve(self, question):
        """使用ReAct框架解决问题"""
        print("🤖 ReAct Framework - Reasoning and Acting")
        print("=" * 60)
        print(f"❓ 问题: {question}")
        print("=" * 60)

        try:
            result = self.agent.invoke({"input": question})
            return result
        except Exception as e:
            return f"执行过程中出现错误: {str(e)}"


# 使用示例
def main():
    # 使用LangChain的ReAct实现
    print("LangChain ReAct Agent")
    print("=" * 80)

    react_framework = ReActFramework()

    # 测试问题
    questions = [
        "埃隆·马斯克现在多大了？他是哪一年出生的？",
        # "计算 23 * 45 + 67 的结果",
        # "小猫爱学休假类型？",
        # "2024年世界杯在哪个国家举办？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n测试 {i}:")
        try:
            result = react_framework.solve(question)
            print(f"✅ 结果: {result}")
        except Exception as e:
            print(f"❌ 错误: {e}")
        print("-" * 60)


if __name__ == "__main__":
    main()