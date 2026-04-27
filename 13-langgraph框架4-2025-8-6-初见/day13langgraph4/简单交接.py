import os
import json
import random
from typing import Literal, TypedDict
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化大模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model='qwen-plus-2025-01-25',
    temperature=0.7
)


def call_llm(system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    """
    真实的LLM调用函数
    """
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        # 设置温度参数
        llm.temperature = temperature
        response = llm.invoke(messages)

        print(f"LLM调用成功")
        print(f"系统提示: {system_prompt[:50]}...")
        print(f"用户输入: {user_message[:50]}...")
        print(f"AI回复: {response.content[:100]}...")

        return response.content

    except Exception as e:
        print(f"LLM调用失败: {str(e)}")
        return f"LLM调用失败: {str(e)}"


class OrderState(TypedDict):
    customer_name: str  # 顾客名称
    order_items: list  # 订单内容
    raw_order_text: str  # 原始订单文本（用户自然语言输入）
    total_amount: float  # 订单总价格
    payment_status: str  # 支付状态
    delivery_address: str  # 配送地址
    order_status: str  # 订单状态
    messages: list  # 消息
    llm_analysis: dict  # LLM分析结果


def order_receiver(state: OrderState) -> Command[Literal["payment_processor", "order_validator"]]:
    """订单接收智能体 - 使用真实大模型理解自然语言订单"""
    raw_text = state.get("raw_order_text", "")
    customer_name = state.get("customer_name", "")
    order_items = state.get("order_items", [])

    print(f"订单接收：{customer_name} 的订单")

    llm_analysis = {}

    # 使用大模型理解和解析自然语言订单
    if raw_text:
        system_prompt = """你是一个专业的餐厅订单理解助手。请仔细分析用户的订单文本，提取以下信息：

1. 识别用户想要的具体食物和饮料
2. 判断订单信息是否清晰完整
3. 识别任何特殊要求或备注
4. 评估订单的明确程度

请严格按照以下JSON格式返回结果：
{
    "items_identified": ["商品1", "商品2"],
    "items_confirmed": true/false,
    "confidence_score": 0.0-1.0,
    "special_requests": "特殊要求描述",
    "clarity_assessment": "订单清晰度评估",
    "suggested_clarification": "如果不清晰，建议询问的问题"
}"""

        llm_response = call_llm(system_prompt, f"用户订单：{raw_text}", temperature=0.3)

        try:
            llm_analysis = json.loads(llm_response)
        except json.JSONDecodeError:
            print("LLM返回的不是有效JSON格式，尝试解析...")
            # 如果不是JSON，创建默认分析结果
            llm_analysis = {
                "items_identified": [],
                "items_confirmed": False,
                "confidence_score": 0.0,
                "clarity_assessment": "解析失败",
                "suggested_clarification": "请重新描述您的订单"
            }

        print(f"LLM分析结果: {llm_analysis}")

        # 基于LLM分析决定流程
        confidence = llm_analysis.get("confidence_score", 0.0)
        items_confirmed = llm_analysis.get("items_confirmed", False)

        if confidence < 0.7 or not items_confirmed:
            return Command(
                goto="order_validator",
                update={
                    "order_status": "需要验证",
                    "llm_analysis": llm_analysis,
                    "messages": [
                        f"订单信息需要确认：{llm_analysis.get('suggested_clarification', '请提供更详细的订单信息')}"]
                }
            )

    # 计算总金额
    total = sum(item.get("price", 0) for item in order_items)

    # 检查订单有效性
    if not order_items or total <= 0:
        return Command(
            goto="order_validator",
            update={
                "order_status": "需要验证",
                "llm_analysis": llm_analysis,
                "messages": ["订单信息不完整，需要验证"]
            }
        )
    else:
        return Command(
            goto="payment_processor",
            update={
                "total_amount": total,
                "order_status": "待支付",
                "llm_analysis": llm_analysis,
                "messages": [f"订单确认成功！总金额：{total}元，准备处理支付"]
            }
        )


def order_validator(state: OrderState) -> Command[Literal["payment_processor", END]]:
    """订单验证智能体 - 使用大模型进行智能验证和建议"""
    print("订单验证：使用AI检查订单完整性")

    order_items = state.get("order_items", [])
    customer_name = state.get("customer_name", "")
    llm_analysis = state.get("llm_analysis", {})
    raw_order_text = state.get("raw_order_text", "")

    # 使用大模型进行订单验证和修正建议
    system_prompt = """你是一个专业的订单验证专家。请分析订单信息，判断：

1. 订单是否有效和完整
2. 是否需要修正或补充信息
3. 给出具体的处理建议
4. 如果订单无效，说明原因

请根据分析结果返回JSON格式：
{
    "is_valid": true/false,
    "validation_score": 0.0-1.0,
    "issues_found": ["问题1", "问题2"],
    "corrections_needed": ["修正建议1", "修正建议2"],
    "processing_recommendation": "继续处理/需要客户确认/取消订单",
    "customer_message": "给客户的反馈消息"
}"""

    order_info = f"""
    客户姓名：{customer_name}
    原始订单文本：{raw_order_text}
    当前订单项：{order_items}
    之前的LLM分析：{llm_analysis}
    """

    llm_response = call_llm(system_prompt, order_info, temperature=0.2)

    try:
        validation_result = json.loads(llm_response)
    except json.JSONDecodeError:
        print("验证结果解析失败")
        validation_result = {
            "is_valid": False,
            "customer_message": "订单验证过程中出现错误，请重新提交订单"
        }

    print(f"验证结果: {validation_result}")

    # 根据验证结果决定流程
    if not validation_result.get("is_valid", False) or validation_result.get("processing_recommendation") == "取消订单":
        return Command(
            goto=END,
            update={
                "order_status": "订单取消",
                "messages": [validation_result.get("customer_message", "订单验证失败，已取消")]
            }
        )

    # 如果需要客户确认但这里简化处理，假设已确认
    corrected_total = sum(item.get("price", 0) for item in order_items) or 50.0

    return Command(
        goto="payment_processor",
        update={
            "total_amount": corrected_total,
            "order_status": "验证通过，待支付",
            "messages": [validation_result.get("customer_message", f"订单验证通过，总金额：{corrected_total}元")]
        }
    )


def payment_processor(state: OrderState) -> Command[Literal["delivery_scheduler", "order_receiver", END]]:
    """支付处理智能体 - 使用大模型生成个性化支付处理消息"""
    total_amount = state.get("total_amount", 0)
    customer_name = state.get("customer_name", "")
    print(f"支付处理：处理 {customer_name} 的 {total_amount} 元支付")

    # 模拟支付处理逻辑
    payment_success = random.choice([True, False, "retry"])

    if payment_success:
        # 使用大模型生成支付成功消息
        system_prompt = f"""请生成一条专业、友好的支付成功确认消息。要求：
1. 感谢客户
2. 确认支付金额
3. 告知下一步流程
4. 保持简洁友好的语调

客户姓名：{customer_name}
支付金额：{total_amount}元"""

        success_message = call_llm(system_prompt, "生成支付成功消息", temperature=0.5)

        return Command(
            goto="delivery_scheduler",
            update={
                "payment_status": "已支付",
                "order_status": "待配送",
                "messages": [success_message]
            }
        )
    elif payment_success == "retry":
        return Command(
            goto="order_receiver",
            update={
                "payment_status": "支付重试",
                "messages": ["支付遇到临时问题，正在重新处理您的订单..."]
            }
        )
    else:
        # 使用大模型生成个性化的支付失败处理消息
        system_prompt = f"""请生成一条专业、同理心的支付失败处理消息。要求：
1. 表达歉意和理解
2. 提供具体的解决方案
3. 给出客服联系方式
4. 保持积极正面的语调
5. 不要让客户感到沮丧

客户姓名：{customer_name}
订单金额：{total_amount}元"""

        failure_message = call_llm(system_prompt, "支付失败，需要友好专业的客服回复", temperature=0.6)

        return Command(
            goto=END,
            update={
                "payment_status": "支付失败",
                "order_status": "订单暂停",
                "messages": [failure_message]
            }
        )


def delivery_scheduler(state: OrderState) -> Command[Literal[END]]:
    """配送安排智能体 - 使用大模型生成个性化配送通知"""
    customer_name = state.get("customer_name", "")
    delivery_address = state.get("delivery_address", "")
    order_items = state.get("order_items", [])
    total_amount = state.get("total_amount", 0)

    print(f"配送安排：为 {customer_name} 安排配送")

    # 🤖 使用大模型生成个性化配送通知
    system_prompt = f"""请生成一条专业、详细的配送通知消息。要求：
1. 个性化问候客户
2. 确认订单详情
3. 提供准确的配送信息
4. 包含联系方式和注意事项
5. 语气友好专业

订单信息：
- 客户：{customer_name}
- 配送地址：{delivery_address}
- 订单商品：{[item.get('name', '未知商品') for item in order_items]}
- 订单金额：{total_amount}元"""

    delivery_notification = call_llm(system_prompt, "生成专业的配送通知消息", temperature=0.4)

    return Command(
        goto=END,
        update={
            "order_status": "配送中",
            "messages": [delivery_notification]
        }
    )


# 构建集成真实大模型的订餐系统
order_builder = StateGraph(OrderState)
order_builder.add_node("order_receiver", order_receiver)
order_builder.add_node("order_validator", order_validator)
order_builder.add_node("payment_processor", payment_processor)
order_builder.add_node("delivery_scheduler", delivery_scheduler)
order_builder.add_edge(START, "order_receiver")

order_system = order_builder.compile()



# 测试运行 - 包含自然语言订单输入
test_order_1 = {
    "customer_name": "张三",
    "raw_order_text": "我想要一个汉堡，再来一杯可乐，谢谢！",
    "order_items": [
        {"name": "汉堡", "price": 25.0},
        {"name": "可乐", "price": 8.0}
    ],
    "delivery_address": "北京市朝阳区xxx街道",
    "messages": [],
    "llm_analysis": {}
}

try:
    result = order_system.invoke(test_order_1)
    print(f"\n最终状态：{result['order_status']}")
    print(f"处理消息：")
    for msg in result['messages']:
        print(f"  - {msg}")
    if result.get('llm_analysis'):
        print(f"LLM分析结果：{result['llm_analysis']}")
except Exception as e:
    print(f"系统执行出错: {str(e)}")