from typing import Literal, Annotated, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import InjectedState, create_react_agent
import json
import os
from dotenv import load_dotenv

load_dotenv()

# 模拟数据
KNOWLEDGE_BASE = {
    "login": "请清除浏览器缓存并重新登录，或重置密码。",
    "payment": "请检查银行卡余额，确认交易状态，或联系银行。",
    "bug": "我们已记录此问题，技术团队将在24小时内处理。",
    "network": "请检查网络连接，或尝试切换网络环境。",
    "performance": "建议清理缓存、重启应用或检查系统资源使用情况。"
}

PRODUCTS = {
    "basic": {"name": "基础版", "price": 99, "features": ["基础功能", "邮件支持"]},
    "pro": {"name": "专业版", "price": 299, "features": ["高级功能", "优先支持", "API访问"]},
    "enterprise": {"name": "企业版", "price": 999, "features": ["企业功能", "专属客服", "定制开发"]}
}

USER_DATABASE = {
    "user123": {"plan": "pro", "status": "active", "support_level": "premium", "balance": 500},
    "user456": {"plan": "basic", "status": "active", "support_level": "standard", "balance": 100}
}

TICKET_SYSTEM = []  # 模拟工单系统


# 扩展状态定义
class CustomerServiceState(MessagesState):
    current_agent: str  # 目前智能体
    customer_id: str  # 用户id
    issue_type: str  # 问题类别
    priority: str  # 优先级
    resolution_status: str  # 解决状态
    routing_reason: str  # 路由原因
    ticket_id: str  # 工单id
    remaining_steps: int  #


# ================================
# 为ReAct Agent定义工具函数
# ================================

@tool
def search_knowledge_base(query: str) -> str:
    """搜索技术知识库，查找解决方案"""
    query_lower = query.lower()

    results = []
    for issue, solution in KNOWLEDGE_BASE.items():
        if issue in query_lower:
            results.append(f"{issue}: {solution}")

    if results:
        return f"找到以下解决方案:\n" + "\n".join(results)
    else:
        return "未在知识库中找到相关解决方案，建议创建技术工单进行人工处理。"


@tool
def get_product_info(product_query: str) -> str:
    """获取产品信息和价格"""
    if not product_query:
        # 返回所有产品信息
        result = "我们的产品线包括:\n\n"
        for key, product in PRODUCTS.items():
            result += f"**{product['name']}** - ¥{product['price']}/月\n"
            result += f"功能: {', '.join(product['features'])}\n\n"
        return result

    # 搜索特定产品
    query_lower = product_query.lower()
    for key, product in PRODUCTS.items():
        if key in query_lower or product['name'] in query_lower:
            return f"**{product['name']}**\n价格: ¥{product['price']}/月\n功能: {', '.join(product['features'])}"

    return f"未找到关于'{product_query}'的产品信息。请查看我们的完整产品列表。"


@tool
def get_user_account_info(user_id: str) -> str:
    """查询用户账户信息"""
    if not user_id:
        return "请提供您的用户ID以查询账户信息。"

    if user_id in USER_DATABASE:
        user_info = USER_DATABASE[user_id]
        return f"""账户信息:
用户ID: {user_id}
当前套餐: {user_info['plan']}
账户状态: {user_info['status']}
支持级别: {user_info['support_level']}
账户余额: ¥{user_info['balance']}"""
    else:
        return f"未找到用户ID '{user_id}' 的账户信息。请检查用户ID是否正确。"


@tool
def create_support_ticket(issue_description: str, priority: str,
                          state: Annotated[Dict, InjectedState]) -> str:
    """创建技术支持工单"""
    import uuid
    import datetime

    ticket_id = f"TICKET-{str(uuid.uuid4())[:8].upper()}"
    customer_id = state.get("customer_id", "anonymous")

    ticket = {
        "id": ticket_id,
        "customer_id": customer_id,
        "issue": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.datetime.now()  # 模拟时间
    }

    TICKET_SYSTEM.append(ticket)

    # 更新状态
    state["ticket_id"] = ticket_id

    return f"已创建支持工单: {ticket_id}\n问题描述: {issue_description}\n优先级: {priority}\n我们的技术团队将在24小时内处理您的问题。"


@tool
def calculate_upgrade_cost(current_plan: str, target_plan: str) -> str:
    """计算升级费用"""
    if current_plan not in PRODUCTS or target_plan not in PRODUCTS:
        return "无效的套餐类型。请检查套餐名称。"

    current_price = PRODUCTS[current_plan]["price"]
    target_price = PRODUCTS[target_plan]["price"]

    if target_price <= current_price:
        return f"目标套餐 ({PRODUCTS[target_plan]['name']}) 价格不高于当前套餐 ({PRODUCTS[current_plan]['name']})，无需升级费用。"

    upgrade_cost = target_price - current_price
    return f"""升级费用计算:
当前套餐: {PRODUCTS[current_plan]['name']} (¥{current_price}/月)
目标套餐: {PRODUCTS[target_plan]['name']} (¥{target_price}/月)
升级费用: ¥{upgrade_cost}/月

新增功能: {', '.join(set(PRODUCTS[target_plan]['features']) - set(PRODUCTS[current_plan]['features']))}"""


@tool
def process_refund_request(user_id: str, reason: str) -> str:
    """处理退款请求"""
    if not user_id or user_id not in USER_DATABASE:
        return "请提供有效的用户ID以处理退款请求。"

    user_info = USER_DATABASE[user_id]
    if user_info["status"] != "active":
        return "只有活跃账户才能申请退款。"

    # 模拟退款处理
    refund_amount = PRODUCTS[user_info["plan"]]["price"]

    return f"""退款申请已提交:
用户ID: {user_id}
退款原因: {reason}
退款金额: ¥{refund_amount}
处理时间: 3-5个工作日
退款将原路返回到您的支付账户。"""


# ================================
# 使用create_react_agent创建专业Agent
# ================================

def create_llm_router_with_react_agents():
    """创建使用ReAct Agent的智能路由系统"""

    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model='qwen-plus-2025-07-14',
        temperature=0.1
    )

    # 1. 创建技术支持ReAct Agent
    tech_tools = [search_knowledge_base, create_support_ticket]
    tech_system_prompt = """你是一个专业的技术支持工程师。你的任务是:

1. 使用search_knowledge_base工具搜索已知问题的解决方案
2. 如果知识库中没有解决方案，使用create_support_ticket创建工单
3. 提供清晰的步骤指导
4. 对于复杂问题，建议用户联系高级技术支持

请始终保持专业、耐心的态度，并确保用户理解每个步骤。"""

    tech_agent = create_react_agent(
        llm,
        tech_tools,
        state_schema=CustomerServiceState,
        prompt=tech_system_prompt
    )

    # 2. 创建销售ReAct Agent
    sales_tools = [get_product_info, calculate_upgrade_cost]
    sales_system_prompt = """你是一个专业的销售顾问。你的任务是:

1. 使用get_product_info工具提供详细的产品信息
2. 使用calculate_upgrade_cost工具帮助客户计算升级费用
3. 根据客户需求推荐合适的产品
4. 解答价格和功能相关问题

请保持热情、专业的态度，关注客户真实需求，提供有价值的建议。"""

    sales_agent = create_react_agent(
        llm,
        sales_tools,
        state_schema=CustomerServiceState,
        prompt=sales_system_prompt
    )

    # 3. 创建客户管理ReAct Agent
    admin_tools = [get_user_account_info, process_refund_request]
    admin_system_prompt = """你是一个客户管理专员。你的任务是:

1. 使用get_user_account_info工具查询客户账户信息
2. 使用process_refund_request工具处理退款申请
3. 处理账户相关问题和权限调整
4. 确保客户数据的安全和隐私

请保持严谨、负责的态度，严格遵守数据保护规定。"""

    admin_agent = create_react_agent(
        llm,
        admin_tools,
        state_schema=CustomerServiceState,
        prompt=admin_system_prompt
    )

    # 4. 创建智能路由器-主管
    def llm_router(state: CustomerServiceState) -> Command[Literal["tech_agent", "sales_agent", "admin_agent", END]]:
        """LLM驱动的智能路由器"""
        messages = state["messages"]
        if not messages:
            return Command(goto=END)

        last_message = messages[-1].content

        routing_prompt = f"""分析用户问题并决定路由到哪个专业部门。

用户问题: {last_message}

部门说明:
- tech_agent: 技术问题、Bug、登录异常、系统错误、性能问题
- sales_agent: 产品咨询、价格询问、功能对比、升级服务、购买流程  
- admin_agent: 账户查询、权限管理、退款申请、账单问题、用户资料

返回JSON格式:
{{
    "route_to": "部门代码",
    "confidence": "置信度(0-1)",
    "reason": "路由原因",
    "priority": "优先级(low/normal/high/urgent)"
}}"""

        try:
            response = llm.invoke([SystemMessage(content=routing_prompt)])
            decision = json.loads(response.content.strip())

            route_to = decision.get("route_to", "tech_agent")
            confidence = decision.get("confidence", 0.8)
            reason = decision.get("reason", "智能路由分析")
            priority = decision.get("priority", "normal")

            print(f"🧠 路由决策: {route_to} | 置信度: {confidence} | 原因: {reason}")

            return Command(
                goto=route_to,
                update={
                    "current_agent": route_to,
                    "issue_type": route_to.replace("_agent", ""),
                    "priority": priority,
                    "routing_reason": reason
                }
            )

        except Exception as e:
            print(f"⚠️ 路由失败，使用默认路由: {e}")
            return Command(
                goto="tech_agent",
                update={
                    "current_agent": "tech_agent",
                    "routing_reason": "路由失败，默认技术支持"
                }
            )

    # 5. 构建图结构
    builder = StateGraph(CustomerServiceState)

    # 添加节点
    builder.add_node("llm_router", llm_router)
    builder.add_node("tech_agent", tech_agent)
    builder.add_node("sales_agent", sales_agent)
    builder.add_node("admin_agent", admin_agent)

    # 设置入口
    builder.add_edge(START, "llm_router")

    return builder.compile()


# ================================
# 纯ReAct Agent架构 - 监督者也是ReAct Agent-工具进行交接
# ================================

def create_pure_react_system():
    """创建纯ReAct Agent架构 - 连监督者都是ReAct Agent"""

    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model='qwen-plus-2025-01-25',
        temperature=0.2
    )

    # 为监督者创建路由工具
    @tool
    def route_to_technical_support(issue_description: str, state: Annotated[dict, InjectedState],
                                   tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """将问题路由到技术支持部门"""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to tech_agent",
            "name": "tech_agent",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto="tech_agent",
            # 在父图中执行导航
            graph=Command.PARENT,
            update={
                "current_agent": "tech_agent",
                "issue_type": "technical",
                "routing_reason": f"技术问题: {issue_description}",
                "messages":
                    state["messages"] + [tool_message]

            }
        )

    @tool
    def route_to_sales_department(inquiry_description: str, state: Annotated[dict, InjectedState],
                                  tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """将咨询路由到销售部门"""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to sales_agent",
            "name": "sales_agent",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto="sales_agent",
            # 在父图中执行导航
            graph=Command.PARENT,
            update={
                "current_agent": "sales_agent",
                "issue_type": "sales",
                "routing_reason": f"销售咨询: {inquiry_description}",
                "messages":
                    state["messages"] + [tool_message]

            }
        )

    @tool
    def route_to_customer_management(request_description: str, state: Annotated[dict, InjectedState],
                                     tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """将请求路由到客户管理部门"""

        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to admin_agent",
            "name": "admin_agent",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto="admin_agent",
            # 在父图中执行导航
            graph=Command.PARENT,
            update={
                "current_agent": "admin_agent",
                "issue_type": "administration",
                "routing_reason": f"销售咨询: {request_description}",
                "messages":
                    state["messages"] + [tool_message]

            }
        )

    # 监督者ReAct Agent
    supervisor_tools = [route_to_technical_support, route_to_sales_department, route_to_customer_management]
    supervisor_prompt = """你是一个智能客服路由监督者。根据用户问题，使用相应的路由工具将问题分配给专业部门:

- route_to_technical_support: 技术问题、系统故障、登录问题、Bug报告
- route_to_sales_department: 产品咨询、价格询问、购买升级、功能对比
- route_to_customer_management: 账户管理、退款申请、权限问题、用户资料

分析用户问题的核心内容，选择最合适的部门处理。一旦路由完成，解释你的决策理由。"""

    supervisor_agent = create_react_agent(
        llm,
        supervisor_tools,
        state_schema=CustomerServiceState,
        prompt=supervisor_prompt
    )

    # 专业部门的ReAct Agent
    tech_tools = [search_knowledge_base, create_support_ticket]
    tech_agent = create_react_agent(
        llm,
        tech_tools,
        state_schema=CustomerServiceState,
        prompt="你是技术支持工程师，使用工具解决技术问题。"
    )

    sales_tools = [get_product_info, calculate_upgrade_cost]
    sales_agent = create_react_agent(
        llm,
        sales_tools,
        state_schema=CustomerServiceState,
        prompt="你是销售顾问，使用工具提供产品信息和升级建议。"
    )

    admin_tools = [get_user_account_info, process_refund_request]
    admin_agent = create_react_agent(
        llm,
        admin_tools,
        state_schema=CustomerServiceState,
        prompt="你是客户管理专员，使用工具处理账户和管理事务。"
    )

    # 构建纯ReAct架构
    builder = StateGraph(CustomerServiceState)
    builder.add_node("supervisor_agent", supervisor_agent)
    builder.add_node("tech_agent", tech_agent)
    builder.add_node("sales_agent", sales_agent)
    builder.add_node("admin_agent", admin_agent)


    builder.add_edge(START, "supervisor_agent")

    return builder.compile()


# ================================
# 测试运行函数
# ================================

def run_react_agent_examples():
    """运行ReAct Agent示例"""

    print("🤖 基于create_react_agent的多智能体客户服务系统")
    print("=" * 70)

    # 测试用例
    test_cases = [
        # {
        #     "message": "我的应用一直崩溃，点击登录按钮就闪退",
        #     "customer_id": "user123"
        # },
        {
            "message": "我想了解专业版和企业版的区别，我们公司大概50人,另外我想查询我的余额，用户ID是user123",
            "customer_id": ""
        },
        # {
        #     "message": "我想查看我的账户余额，用户ID是user123，另外我想申请退款",
        #     "customer_id": "user123"
        # },
        # {
        #     "message": "系统报错500，数据库连接失败，这是生产环境的紧急问题",
        #     "customer_id": "user456"
        # }
    ]

    systems = [
        # ("混合架构 (LLM路由 + ReAct Agent)", create_llm_router_with_react_agents),  # 基础交接方式
        ("纯ReAct架构 (监督者也是ReAct Agent)", create_pure_react_system)  # 工具交接方式
    ]

    for system_name, create_system in systems:
        print(f"\n🔥 {system_name}")
        print("=" * 70)

        try:
            graph = create_system()


            for i, test_case in enumerate(test_cases, 1):
                print(f"\n--- 测试案例 {i} ---")
                print(f"用户: {test_case['message']}")
                print("-" * 50)

                initial_state = {
                    "messages": [HumanMessage(content=test_case['message'])],
                    "current_agent": "start",
                    "customer_id": test_case['customer_id'],
                    "issue_type": "",
                    "priority": "normal",
                    "resolution_status": "pending",
                    "routing_reason": "",
                    "ticket_id": "",
                    "remaining_steps": 10
                }

                try:
                    result = graph.invoke(initial_state)

                    # 显示AI回复
                    print("🤖 AI回复:")
                    for msg in result["messages"]:
                        if isinstance(msg, AIMessage):
                            # 截取长回复以保持可读性
                            content = msg.content
                            if len(content) > 300:
                                content = content[:300] + "..."
                            print(f"  {content}")

                    # 显示处理信息
                    print(f"\n📊 处理状态:")
                    print(f"  处理部门: {result.get('current_agent', 'N/A')}")
                    print(f"  问题类型: {result.get('issue_type', 'N/A')}")
                    print(f"  优先级: {result.get('priority', 'N/A')}")
                    if result.get('ticket_id'):
                        print(f"  工单ID: {result.get('ticket_id')}")

                except Exception as e:
                    print(f"❌ 处理失败: {e}")

        except Exception as e:
            print(f"❌ 系统创建失败: {e}")


if __name__ == "__main__":
    # 检查API配置
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️  请设置DASHSCOPE_API_KEY环境变量")
        print("export DASHSCOPE_API_KEY='your-api-key-here'")
    else:
        run_react_agent_examples()

        # 显示工单系统状态
        if TICKET_SYSTEM:
            print(f"\n📋 创建的工单 ({len(TICKET_SYSTEM)} 个):")
            for ticket in TICKET_SYSTEM:
                print(f"  {ticket['id']}: {ticket['issue'][:50]}...")