from typing import TypedDict
from langgraph.graph import StateGraph, END

# 定义状态
class MyState(TypedDict):
    user_type: str  # "vip", "normal", "guest"
    message: str  # 用户消息
    result: str  # 返回内容


# 定义不同的处理节点
def vip_service(state):
    """VIP 用户服务"""
    return {"result": f"VIP专享服务: {state['message']}"}


def normal_service(state):
    """普通用户服务"""
    return {"result": f"标准服务: {state['message']}"}


def guest_service(state):
    """游客服务"""
    return {"result": f"游客服务(功能受限): {state['message']}"}


# 条件入口点函数
def route_by_user_type(state):
    """根据用户类型路由到不同的服务"""
    user_type = state["user_type"]

    if user_type == "vip":
        return "vip_service"
    elif user_type == "normal":
        return "normal_service"
    else:
        return "guest_service"


# 构建图
workflow = StateGraph(state_schema=MyState)

# 添加节点
workflow.add_node("vip_service", vip_service)
workflow.add_node("normal_service", normal_service)
workflow.add_node("guest_service", guest_service)

# 设置条件入口点 - 关键部分！
workflow.set_conditional_entry_point(
    route_by_user_type,  # 条件函数
    # 会根据value值找到对应的节点
    {
        "vip_service": "vip_service",
        "normal_service": "normal_service",
        "guest_service": "guest_service"
    }
)

# 添加结束边
workflow.add_edge("vip_service", END)
workflow.add_edge("normal_service", END)
workflow.add_edge("guest_service", END)

# 编译图
app = workflow.compile()
# 测试 VIP 用户
result1 = app.invoke({
    "user_type": "vip",
    "message": "我要退款",
    "result": ""
})
print("VIP用户:", result1)