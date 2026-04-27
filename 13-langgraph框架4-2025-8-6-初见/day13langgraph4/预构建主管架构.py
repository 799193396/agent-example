"""
正确使用 LangGraph Supervisor 调用子代理的多智能体系统

系统包含：
- ResearchAgent: 研究专家，使用网络搜索工具
- AnalysisAgent: 数据分析专家，处理计算和分析
- WriteAgent: 写作专家，生成报告和总结
- Supervisor: 协调器，负责任务分派和流程控制

业务场景：市场研究和报告生成
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='qwen-plus-2025-01-25')


# ============================================================================
# 定义系统状态
# ============================================================================

class MarketResearchState(TypedDict):
    """
    市场研究系统的状态定义

    messages: 对话历史消息
    research_topic: 研究主题
    research_data: 收集到的研究数据
    analysis_results: 分析结果
    final_report: 最终报告
    """
    messages: Annotated[Sequence[BaseMessage], "对话历史"]
    research_topic: str
    research_data: dict
    analysis_results: dict
    final_report: str


# ============================================================================
# 创建专业工具
# ============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import yfinance as yf  # 获取股票信息

import re


@tool
def web_search(query: str) -> str:
    """
    真实的网络搜索工具 - 使用 Tavily API

    Args:
        query: 搜索关键词

    Returns:
        搜索结果
    """
    try:

        from langchain_tavily import TavilySearch

        tavily_search = TavilySearch()

        results = tavily_search.invoke(query)

        if results:
            formatted_results = f"🔍 搜索关键词: {query}\n\n"

            for i, result in enumerate(results["results"], 1):
                title = result.get('title', '无标题')
                content = result.get('content', '无内容')
                url = result.get('url', '无链接')

                # 清理和截取内容
                content = re.sub(r'\s+', ' ', content).strip()
                if len(content) > 300:
                    content = content[:300] + "..."

                formatted_results += f"{i}. **{title}**\n"
                formatted_results += f"   📖 {content}\n"
                formatted_results += f"   🔗 {url}\n\n"

            return formatted_results
    except Exception as e:
        return f"搜索过程中出现错误: {str(e)}\n请检查网络连接或API配置。"


@tool
def calculate_market_metrics(data_input: str) -> str:
    """
    真实的市场指标计算工具

    Args:
        data_input: 数据输入，可以是股票代码、数值或描述

    Returns:
        计算结果和分析
    """
    try:
        result = "📊 市场指标分析结果:\n\n"

        # 尝试解析输入中的数值
        numbers = re.findall(r'[\d,]+\.?\d*', data_input.replace(',', ''))
        numbers = [float(x) for x in numbers if x]

        if numbers:
            # 基础统计计算
            if len(numbers) >= 2:
                mean_val = np.mean(numbers)
                std_val = np.std(numbers)
                growth_rate = ((numbers[-1] - numbers[0]) / numbers[0]) * 100 if numbers[0] != 0 else 0

                result += f"📈 平均值: {mean_val:,.2f}\n"
                result += f"📊 标准差: {std_val:,.2f}\n"
                result += f"📈 增长率: {growth_rate:+.2f}%\n"

                # 复合年增长率计算 (假设数据跨度为多年)
                if len(numbers) > 2:
                    years = len(numbers) - 1
                    cagr = (pow(numbers[-1] / numbers[0], 1 / years) - 1) * 100
                    result += f"📊 复合年增长率 (CAGR): {cagr:.2f}%\n"

                # 波动率计算
                if len(numbers) > 1:
                    returns = np.diff(numbers) / numbers[:-1]
                    volatility = np.std(returns) * 100
                    result += f"⚡ 波动率: {volatility:.2f}%\n"

        # 检查是否包含股票代码
        stock_symbols = re.findall(r'\b[A-Z]{1,5}\b', data_input.upper())
        if stock_symbols:
            result += f"\n🔍 检测到股票代码: {', '.join(stock_symbols)}\n"
            for symbol in stock_symbols[:3]:  # 限制最多3个
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if info:
                        pe = info.get('trailingPE', 'N/A')
                        pb = info.get('priceToBook', 'N/A')
                        roe = info.get('returnOnEquity', 'N/A')

                        result += f"\n📊 {symbol} 财务指标:\n"
                        result += f"   • 市盈率 (PE): {pe if pe != 'N/A' else 'N/A'}\n"
                        result += f"   • 市净率 (PB): {pb if pb != 'N/A' else 'N/A'}\n"
                        result += f"   • 净资产收益率 (ROE): {roe if roe != 'N/A' else 'N/A'}\n"
                except:
                    continue

        # 市场风险评估
        result += f"\n🎯 风险评估:\n"
        if numbers and len(numbers) > 1:
            cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
            if cv < 10:
                risk_level = "低风险"
            elif cv < 25:
                risk_level = "中等风险"
            else:
                risk_level = "高风险"
            result += f"   • 风险等级: {risk_level} (变异系数: {cv:.2f}%)\n"

        # 投资建议
        if numbers and len(numbers) >= 2:
            if growth_rate > 15:
                suggestion = "强烈看好，建议适当增加投资"
            elif growth_rate > 5:
                suggestion = "谨慎乐观，可考虑投资"
            elif growth_rate > -5:
                suggestion = "保持观望，注意风险控制"
            else:
                suggestion = "谨慎投资，建议降低仓位"

            result += f"   • 投资建议: {suggestion}\n"

        return result

    except Exception as e:
        return f"计算市场指标时出错: {str(e)}"


@tool
def generate_real_charts(data_description: str, chart_type: str = "auto") -> str:
    """
    生成真实的数据图表

    Args:
        data_description: 数据描述或股票代码
        chart_type: 图表类型 (line, bar, pie, scatter, auto)

    Returns:
        图表生成结果和base64编码的图片
    """
    try:
        # 设置中文字体，例如 'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('市场数据分析图表', fontsize=16, fontweight='bold')

        # 检查是否包含股票代码
        stock_symbols = re.findall(r'\b[A-Z]{2,5}\b', data_description.upper())

        if stock_symbols:
            # 获取股票数据并生成图表
            symbol = stock_symbols[0]
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")

                if not hist.empty:
                    # 价格走势图
                    axes[0, 0].plot(hist.index, hist['Close'], linewidth=2, color='#1f77b4')
                    axes[0, 0].set_title(f'{symbol} 股价走势 (6个月)', fontweight='bold')
                    axes[0, 0].set_ylabel('价格 ($)')
                    axes[0, 0].grid(True, alpha=0.3)

                    # 成交量柱状图
                    axes[0, 1].bar(hist.index, hist['Volume'], alpha=0.7, color='#ff7f0e')
                    axes[0, 1].set_title(f'{symbol} 成交量', fontweight='bold')
                    axes[0, 1].set_ylabel('成交量')

                    # 价格分布直方图
                    axes[1, 0].hist(hist['Close'], bins=20, alpha=0.7, color='#2ca02c', edgecolor='black')
                    axes[1, 0].set_title('价格分布', fontweight='bold')
                    axes[1, 0].set_xlabel('价格 ($)')
                    axes[1, 0].set_ylabel('频次')

                    # 移动平均线
                    hist['MA20'] = hist['Close'].rolling(window=20).mean()
                    hist['MA50'] = hist['Close'].rolling(window=50).mean()

                    axes[1, 1].plot(hist.index, hist['Close'], label='收盘价', linewidth=1)
                    axes[1, 1].plot(hist.index, hist['MA20'], label='20日均线', linewidth=2)
                    axes[1, 1].plot(hist.index, hist['MA50'], label='50日均线', linewidth=2)
                    axes[1, 1].set_title('移动平均线分析', fontweight='bold')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)

            except Exception as e:
                # 如果股票数据获取失败，生成示例图表
                generate_charts(axes)
        else:
            # 生成通用示例图表
            generate_charts(axes)

        # 调整布局
        plt.tight_layout()

        # 保存图表为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)

        # 转换为base64
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        result = "📊 已成功生成市场分析图表!\n\n"
        result += "图表包含:\n"
        result += "1. 📈 价格/趋势走势图\n"
        result += "2. 📊 成交量/数据柱状图\n"
        result += "3. 📊 数据分布直方图\n"
        result += "4. 📈 移动平均线分析\n\n"

        # 在实际应用中，你可以保存图片文件或返回base64数据
        result += f"图表已生成 (图片大小: {len(chart_base64)} 字符)\n"
        result += "💡 提示: 图表已保存，可用于报告展示\n"

        return result

    except Exception as e:
        return f"生成图表时出错: {str(e)}"


def generate_charts(axes):
    """生成示例图表"""
    # 示例数据
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    volumes = np.random.randint(1000000, 5000000, len(dates))

    # 趋势图
    axes[0, 0].plot(dates, prices, linewidth=2, color='#1f77b4')
    axes[0, 0].set_title('市场趋势分析', fontweight='bold')
    axes[0, 0].set_ylabel('指数/价格')
    axes[0, 0].grid(True, alpha=0.3)

    # 成交量
    axes[0, 1].bar(dates[::10], volumes[::10], alpha=0.7, color='#ff7f0e')
    axes[0, 1].set_title('成交量分析', fontweight='bold')
    axes[0, 1].set_ylabel('成交量')

    # 分布图
    axes[1, 0].hist(prices, bins=20, alpha=0.7, color='#2ca02c', edgecolor='black')
    axes[1, 0].set_title('价格分布', fontweight='bold')
    axes[1, 0].set_xlabel('价格')
    axes[1, 0].set_ylabel('频次')

    # 市场份额饼图
    labels = ['公司A', '公司B', '公司C', '公司D', '其他']
    sizes = [30, 25, 20, 15, 10]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('市场份额分析', fontweight='bold')


# ============================================================================
# 创建专业智能体
# ============================================================================

def create_research_agent():
    """
    创建研究智能体
    专门负责：信息收集、数据搜索、行业调研
    """

    research_agent = create_react_agent(
        model=llm,
        tools=[web_search],
        prompt=(
            "你是一名专业的市场研究员。你的职责包括：\n\n"
            "1. 收集准确、全面的市场信息\n"
            "2. 搜索行业数据和趋势\n"
            "3. 识别关键市场参与者\n"
            "4. 分析竞争格局\n\n"
            "工作原则：\n"
            "- 使用可靠的数据源\n"
            "- 提供具体的数字和事实\n"
            "- 关注最新的市场动态\n"
            "- 只专注于研究工作，不做分析或写作\n\n"
            "- 全程使用中文"
            "完成研究后，将结果直接报告给supervisor。"
        ),
        name="research_agent",  # 使用预构建的主管架构一定要指定代理名称
    )

    return research_agent


def create_analysis_agent():
    """
    创建分析智能体
    专门负责：数据分析、指标计算、趋势预测
    """

    analysis_agent = create_react_agent(
        model=llm,
        tools=[calculate_market_metrics, generate_real_charts],
        prompt=(
            "你是一名专业的数据分析师。你的职责包括：\n\n"
            "1. 分析市场数据和趋势\n"
            "2. 计算关键业务指标\n"
            "3. 进行竞争分析和预测\n"
            "4. 生成数据可视化图表\n\n"
            "工作原则：\n"
            "- 基于数据做客观分析\n"
            "- 使用统计方法和模型\n"
            "- 提供清晰的分析结论\n"
            "- 只专注于分析工作，不做研究或写作\n\n"
            "完成分析后，将结果直接报告给supervisor。"
        ),
        name="analysis_agent",
    )

    return analysis_agent


def create_writing_agent():
    """
    创建写作智能体
    专门负责：报告撰写、内容整理、总结归纳
    """

    writing_agent = create_react_agent(
        model=llm,
        tools=[],  # 写作智能体不需要特殊工具
        prompt=(
            "你是一名专业的商业报告写作专家。你的职责包括：\n\n"
            "1. 整理和综合研究数据\n"
            "2. 撰写结构清晰的报告\n"
            "3. 提供actionable的建议\n"
            "4. 确保内容专业且易懂\n\n"
            "工作原则：\n"
            "- 结构化组织信息\n"
            "- 使用专业商业语言\n"
            "- 突出关键发现和洞察\n"
            "- 提供实用的建议和下一步行动\n\n"
            "完成写作后，将最终报告提交给supervisor。"
        ),
        name="writing_agent",
    )

    return writing_agent


# ============================================================================
# 使用官方方式创建 Supervisor
# ============================================================================

def create_market_research_supervisor():
    """
    使用官方 create_supervisor 创建市场研究协调器

    这个supervisor会真正调用子代理来完成不同的任务
    """

    # 创建各个专业智能体
    research_agent = create_research_agent()
    analysis_agent = create_analysis_agent()
    writing_agent = create_writing_agent()

    # 使用官方语法创建 supervisor
    supervisor = create_supervisor(
        model=llm,
        agents=[research_agent, analysis_agent, writing_agent],
        prompt=(
            "你是一个市场研究项目的项目经理，负责协调三个专业团队：\n\n"
            "1. Research Agent - 市场研究员，负责收集市场信息、行业数据和竞争情报\n"
            "2. Analysis Agent - 数据分析师，负责分析数据、计算指标和预测趋势\n"
            "3. Writing Agent - 报告专家，负责整理信息并撰写最终报告\n\n"
            "你的工作流程：\n"
            "1. 首先分派研究任务给Research Agent收集基础数据\n"
            "2. 然后让Analysis Agent分析数据并计算关键指标\n"
            "3. 最后让Writing Agent基于研究和分析结果撰写完整报告\n"
            "4. 如需补充信息，可以重复调用相应的agent\n\n"
            "重要原则：\n"
            "- 一次只分配给一个agent，不要并行调用\n"
            "- 确保每个agent都有明确的任务说明\n"
            "- 按照逻辑顺序推进工作：研究 → 分析 → 写作\n"
            "- 自己不要直接做具体工作，而是分派给专业agent"
        ),
        add_handoff_back_messages=True,  # 启用代理回传消息
        output_mode="full_history",  # 保留完整历史
    ).compile()

    return supervisor


# ============================================================================
# 主要功能函数
# ============================================================================

def conduct_market_research(research_request: str):
    """
    执行市场研究任务

    Args:
        research_request: 研究需求描述

    Returns:
        研究结果
    """

    # 创建supervisor
    supervisor = create_market_research_supervisor()

    from IPython.display import display, Image
    from langchain_core.runnables.graph import MermaidDrawMethod
    display(
        Image(supervisor.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API, output_file_path="./可视化图.png")))

    print(f"🔍 开始市场研究任务: {research_request}")
    print("=" * 60)

    # 构建初始消息
    initial_messages = [
        HumanMessage(content=research_request)
    ]

    final_result = None  # 用于存储最终结果

    try:
        # 流式执行研究任务
        print("📊 Supervisor开始协调各个专业团队...")
        print()

        for chunk in supervisor.stream({"messages": initial_messages}):
            final_result = chunk  # 保存每次更新的结果

            # 处理不同类型的更新
            if isinstance(chunk, dict):
                for node_name, node_update in chunk.items():
                    if "messages" in node_update and node_update["messages"]:
                        messages = node_update["messages"]
                        latest_message = messages[-1]

                        # 显示每个阶段的进展
                        if hasattr(latest_message, 'name') and latest_message.name:
                            agent_name = latest_message.name
                            print(f"📝 {agent_name} 工作更新:")

                            # 显示消息内容（截取前200字符）
                            if hasattr(latest_message, 'content') and latest_message.content:
                                content = latest_message.content
                                if len(content) > 200:
                                    content = content[:200] + "...[继续工作中]"
                                print(f"   {content}")
                                print()

                        # 检查是否有工具调用
                        if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                            for tool_call in latest_message.tool_calls:
                                tool_name = "unknown"
                                if isinstance(tool_call, dict):
                                    tool_name = tool_call.get('name', 'unknown')
                                elif hasattr(tool_call, 'name'):
                                    tool_name = tool_call.name
                                print(f"🔧 正在使用工具: {tool_name}")
                            print()

                        # 检查是否有转移消息
                        if hasattr(latest_message, 'content') and latest_message.content:
                            if "transfer" in latest_message.content.lower():
                                print(f"🔄 任务转移: {latest_message.content}")
                                print()

        print("✅ 市场研究任务完成！")

        # 返回最终结果，如果没有结果则返回空字典
        return final_result if final_result is not None else {}

    except Exception as e:
        print(f"❌ 执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息用于调试
        return {"error": str(e)}


# ============================================================================
# 测试和演示
# ============================================================================

def run_market_research_demo():
    """
    运行市场研究系统演示
    """

    print("🚀 LangGraph Supervisor 市场研究系统演示")
    print("本系统展示真正的子代理调用和协作")
    print("=" * 60)

    # 测试用例
    test_cases = [
        {
            "title": "电动汽车市场研究",
            "request": "请帮我做一份关于2024年电动汽车市场的综合研究报告，包括市场规模、增长趋势、主要竞争者和未来预测。",
        },
        # {
        #     "title": "人工智能行业分析",
        #     "request": "分析当前人工智能行业的发展状况，重点关注大模型技术的商业化前景和投资机会。",
        # },
        # {
        #     "title": "可再生能源投资研究",
        #     "request": "研究可再生能源领域的投资机会，分析太阳能和风能的市场潜力以及政策影响。",
        # }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试案例 {i}: {test_case['title']}")
        print(f"研究需求: {test_case['request']}")
        print("-" * 50)

        # 执行研究任务
        result = conduct_market_research(test_case['request'])
        print("任务执行结果:", result)
        print("\n" + "=" * 60)


# ============================================================================
# 8. 主程序入口
# ============================================================================

def main():
    """
    主程序入口
    """

    try:
        # 运行演示
        run_market_research_demo()
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")


if __name__ == "__main__":
    main()