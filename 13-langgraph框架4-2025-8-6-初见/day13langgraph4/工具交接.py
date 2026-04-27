from typing import Annotated
from typing_extensions import Literal

from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage, convert_to_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph, START, END

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化大模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model='qwen-plus-2025-07-14',
    temperature=0.7
)


def make_handoff_tool(*, agent_name: str):
    """
    创建一个工具交接函数，用于在代理之间进行转接

    Args:
        agent_name (str): 目标代理的名称

    Returns:
        tool: 返回一个可以执行代理转接的工具函数
    """
    # 根据目标代理名称动态生成工具名称
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
            # 注入当前图状态（LLM会忽略此参数，但工具内部可以使用）
            state: Annotated[dict, InjectedState],
            # 注入当前工具调用ID（LLM会忽略此参数，用于工具消息追踪）
            tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """请求另一个代理的帮助进行任务交接"""

        # 创建工具响应消息，表示成功转接到目标代理
        tool_message = {
            "role": "tool",
            "content": f"成功转接到 {agent_name} 代理",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }

        # 返回Command对象，用于导航到父图中的另一个代理节点
        return Command(
            # 导航到目标代理节点
            goto=agent_name,
            # 在父图中执行导航
            graph=Command.PARENT,
            # 更新状态：将完整的消息历史传递给目标代理，并添加工具消息
            # 这确保了聊天历史的完整性和有效性
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent


# 其实本质就是定义了一个function call执行流程的图
def make_agent(model, tools, system_prompt=None):
    """
    创建一个智能代理，能够使用工具并在需要时进行代理转接

    Args:
        model: 语言模型实例
        tools: 代理可用的工具列表
        system_prompt: 系统提示词，定义代理的角色和行为

    Returns:
        compiled_graph: 编译后的代理图
    """
    # 将工具绑定到模型上
    model_with_tools = model.bind_tools(tools)

    # 创建工具名称到工具对象的映射，便于快速查找
    tools_by_name = {tool.name: tool for tool in tools}

    def call_model(state: MessagesState) -> Command[Literal["call_tools", END]]:
        """
        调用语言模型生成响应

        Args:
            state: 当前消息状态

        Returns:
            Command: 如果需要调用工具则转到call_tools，否则结束
        """
        # 获取对应用户的问题
        messages = state["messages"]

        # 如果有系统提示词，将其添加到消息开头
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # 调用绑定了工具的模型==》包含了要调用工具的数据
        response = model_with_tools.invoke(messages)

        # 检查模型是否决定使用工具
        if len(response.tool_calls) > 0:
            # 如果有工具调用，转到工具执行节点
            return Command(goto="call_tools", update={"messages": [response]})

        # 如果没有工具调用，直接返回响应消息
        return {"messages": [response]}

    def call_tools(state: MessagesState) -> Command[Literal["call_model"]]:
        """
        执行工具调用

        Args:
            state: 当前消息状态

        Returns:
            list[Command]: 工具执行结果的命令列表
        """
        # 获取最后一条消息中的所有工具调用
        tool_calls = state["messages"][-1].tool_calls
        results = []

        # 逐个执行工具调用
        for tool_call in tool_calls:
            # 根据工具名称获取对应的工具对象
            tool_ = tools_by_name[tool_call["name"]]

            # 获取工具的输入参数架构
            tool_input_fields = tool_.get_input_schema().model_json_schema()[
                "properties"
            ]

            # 检查工具是否需要状态注入（简化版实现）
            if "state" in tool_input_fields:
                # 如果工具需要状态，将当前状态注入到工具参数中
                tool_call = {**tool_call, "args": {**tool_call["args"], "state": state}}

            # 执行工具调用
            tool_response = tool_.invoke(tool_call)

            # 处理不同类型的工具响应
            if isinstance(tool_response, ToolMessage):
                # 标准工具消息响应
                results.append(Command(update={"messages": [tool_response]}))
            elif isinstance(tool_response, Command):
                # 直接返回Command对象的工具（如转接工具）
                results.append(tool_response)
            else:
                # 普通响应，转换为工具消息
                tool_message = ToolMessage(
                    content=str(tool_response),
                    tool_call_id=tool_call["id"]
                )
                results.append(Command(update={"messages": [tool_message]}))

        # 返回所有工具执行结果
        return results

    # 构建代理的内部图结构
    graph = StateGraph(MessagesState)

    # 添加模型调用节点和工具调用节点
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", call_tools)

    # 设置图的边：从开始到模型调用，从工具调用回到模型调用
    graph.add_edge(START, "call_model")
    graph.add_edge("call_tools", "call_model")

    # 编译并返回图
    return graph.compile()


def pretty_print_messages(update):
    """
    美化打印消息更新，用于调试和展示

    Args:
        update: 图更新信息，可能是元组或字典
    """
    # 检查是否是来自子图的更新
    if isinstance(update, tuple):
        ns, update = update

        # 跳过父图更新的打印
        if len(ns) == 0:
            return

        # 提取子图ID并打印
        graph_id = ns[-1].split(":")[0]
        print(f"来自子图 {graph_id} 的更新:")
        print()

    # 遍历所有节点更新
    for node_name, node_update in update.items():
        print(f"来自节点 {node_name} 的更新:")
        print()

        # 美化打印所有消息
        if "messages" in node_update:
            for m in convert_to_messages(node_update["messages"]):
                m.pretty_print()
        print()


# ============= 定义数学工具 =============

@tool
def add(a: int, b: int) -> int:
    """执行两个数字的加法运算"""
    result = a + b
    print(f"执行加法: {a} + {b} = {result}")
    return result


@tool
def multiply(a: int, b: int) -> int:
    """执行两个数字的乘法运算"""
    result = a * b
    print(f"执行乘法: {a} × {b} = {result}")
    return result


@tool
def subtract(a: int, b: int) -> int:
    """执行两个数字的减法运算"""
    result = a - b
    print(f"执行减法: {a} - {b} = {result}")
    return result


@tool
def divide(a: int, b: int) -> float:
    """执行两个数字的除法运算"""
    if b == 0:
        return "错误：不能除以零"
    result = a / b
    print(f"执行除法: {a} ÷ {b} = {result}")
    return result


# ============= 演示单个代理 =============

def demo_single_agent():
    """演示单个具有所有数学工具的代理"""
    print("=" * 60)
    print("演示：单个数学代理")
    print("=" * 60)

    # 创建一个拥有所有数学工具的代理
    math_agent = make_agent(
        llm,
        [add, multiply, subtract, divide],
        system_prompt="你是一个数学专家，可以执行各种数学运算。请一步步解决问题。"
    )

    print("问题: 计算 (3 + 5) × 12")
    print()

    # 运行代理并显示结果
    for chunk in math_agent.stream({"messages": [("user", "计算 (3 + 5) × 12")]}):
        pretty_print_messages(chunk)


# ============= 演示多代理协作 =============

def demo_multi_agent_collaboration():
    """演示多个专业代理之间的协作"""
    print("=" * 60)
    print("演示：多代理协作系统")
    print("=" * 60)

    # 创建加法专家代理
    addition_expert = make_agent(
        llm,
        [add, subtract, make_handoff_tool(agent_name="multiplication_expert")],
        system_prompt="""你是加法和减法专家。你精通加法和减法运算。
            当你完成加法或减法运算后，如果后续还需要乘法或除法运算，请转接给乘法专家。
            不要尝试自己完成乘法运算。
            """
    )

    # 创建乘法专家代理
    multiplication_expert = make_agent(
        llm,
        [multiply, divide, make_handoff_tool(agent_name="addition_expert")],
        system_prompt="""你是乘法和除法专家。你精通乘法和除法运算。
            当你接收到需要乘法运算的任务时，请立即执行乘法运算。
            如果后续还需要加法或减法运算，请转接给加法专家。
            当前任务：执行乘法运算并给出最终答案。"""
    )

    # 构建多代理协作图
    builder = StateGraph(MessagesState)

    # 添加两个专家代理节点
    builder.add_node("addition_expert", addition_expert)
    builder.add_node("multiplication_expert", multiplication_expert)

    # 设置入口点为加法专家
    builder.add_edge(START, "addition_expert")

    # 编译协作图
    collaboration_graph = builder.compile()

    print("问题: 计算 (3 + 5) × 12")
    print("加法专家将处理加法，然后转接给乘法专家处理乘法")
    print()

    # 运行协作图并显示子图中的所有更新
    for chunk in collaboration_graph.stream(
            {"messages": [("user", "请计算 (3 + 5) × 12")]},
            subgraphs=True  # 包含子图更新
    ):
        pretty_print_messages(chunk)

    from IPython.display import Image, display
    from langchain_core.runnables.graph import MermaidDrawMethod
    display(
        Image(
            collaboration_graph.get_graph().draw_mermaid_png(
                output_file_path='./工具交接.png'
            )
        )
    )


# ============= 更复杂的协作示例 =============

def demo_complex_collaboration():
    """演示更复杂的多步骤协作"""
    print("=" * 60)
    print("演示：复杂多步协作")
    print("=" * 60)

    # 创建基础运算专家
    basic_math_expert = make_agent(
        llm,
        [add, subtract, make_handoff_tool(agent_name="advanced_math_expert")],
        system_prompt="""你是基础数学专家，专门处理加法和减法。
        对于乘法、除法等高级运算，请转接给高级数学专家。
        你需要先处理括号内的基础运算。"""
    )

    # 创建高级运算专家
    advanced_math_expert = make_agent(
        llm,
        [multiply, divide, make_handoff_tool(agent_name="basic_math_expert")],
        system_prompt="""你是高级数学专家，专门处理乘法和除法。
        对于加法、减法等基础运算，请转接给基础数学专家。
        你负责处理复杂的乘除运算。"""
    )

    # 构建协作图
    builder = StateGraph(MessagesState)
    builder.add_node("basic_math_expert", basic_math_expert)
    builder.add_node("advanced_math_expert", advanced_math_expert)
    builder.add_edge(START, "basic_math_expert")

    complex_graph = builder.compile()

    print("复杂问题: 计算 ((10 + 5) × 3 - 8) ÷ 2")
    print("将需要多次代理转接来完成计算")
    print()

    for chunk in complex_graph.stream(
            {"messages": [("user", "请逐步计算 ((10 + 5) × 3 - 8) ÷ 2")]},
            subgraphs=True
    ):
        pretty_print_messages(chunk)


# ============= 主程序入口 =============

def main():
    """主程序，运行所有演示"""
    print("LangGraph工具交接案例演示")
    print("展示单代理和多代理协作的数学计算系统")
    print()

    try:
        # 演示1：单个代理
        # demo_single_agent()
        #
        # print("\n" + "-" * 20 + "\n")

        # 演示2：多代理协作
        demo_multi_agent_collaboration()

        # print("\n" + "-" * 20 + "\n")
        #
        # # 演示3：复杂协作
        # demo_complex_collaboration()

    except Exception as e:
        print(f"运行出错: {e}")


if __name__ == "__main__":
    main()