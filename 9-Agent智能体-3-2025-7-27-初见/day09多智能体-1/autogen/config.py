import os
from dotenv import load_dotenv
load_dotenv()
# API配置
CONFIG_LIST = [
    {
        "model": "qwen-plus-2025-01-25",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),  # 请替换为您的API密钥
        "base_url": os.getenv("DASHSCOPE_BASE_URL"),
    }
]

# LLM配置
LLM_CONFIG = {
    "config_list": CONFIG_LIST,
    "temperature": 0.7,
    "timeout": 120,
}

# 智能体配置
AGENT_CONFIG = {
    "user_proxy": {
        "name": "UserProxy",
        "system_message": """您是用户代表，负责：
        1. 接收用户需求并转达给其他智能体
        2. 对任务结果进行确认和反馈
        3. 决定是否需要进一步优化
        """,
        "human_input_mode": "NEVER",  # 是否启用人机交互
        "max_consecutive_auto_reply": 5,  # 对话长度限制
        "code_execution_config": {
            "work_dir": "coding_output",
            "use_docker": False,
        },
    },

    "assistant": {
        "name": "CodingAssistant",
        "system_message": """您是专业的编程助手，负责：
        1. 理解和分析编程任务需求
        2. 编写高质量的代码
        3. 提供详细的代码说明和注释
        4. 确保代码的可执行性和安全性

        请始终提供完整、可运行的代码解决方案。
        """,
    },

    "monitor": {
        "name": "CodeReviewer",
        "system_message": """您是代码审查专家，负责：
        1. 检查代码质量和规范性
        2. 识别潜在的bug和安全问题
        3. 提出改进建议
        4. 确保代码符合最佳实践

        对每个代码方案都要进行严格审查，并给出明确的通过/修改建议。
        """,
    },

    "tester": {
        "name": "Tester",
        "system_message": """您是测试工程师，负责：
        1. 设计和编写测试用例
        2. 验证代码功能的正确性
        3. 测试边界条件和异常情况
        4. 提供测试报告和建议
        """,
    }
}

# 对话配置
CHAT_CONFIG = {
    "max_round": 2,  # 课后可以把这个值改成10，让所有的Agent进行工作
    "allow_repeat_speaker": False,
    "manager_system_message": """您是多智能体协作的管理者，负责：
    1. 协调各个智能体的对话顺序
    2. 确保任务按既定流程进行
    3. 监控任务完成质量
    4. 在适当时机终止对话
    """
}

# 任务示例配置
SAMPLE_TASKS = {
    "task1": """
    请创建一个Python函数，用于处理CSV文件：
    1. 读取CSV文件
    2. 计算数值列的统计信息（平均值、最大值、最小值）
    3. 保存处理结果到新的CSV文件
    4. 包含适当的错误处理
    """,

    "task2": """
    创建一个简单的Flask Web应用：
    1. 包含主页和关于页面
    2. 使用Bootstrap美化界面
    3. 实现一个简单的表单提交功能
    4. 包含基本的输入验证
    """,

    "task3": """
    使用matplotlib创建数据可视化脚本：
    1. 生成模拟的销售数据
    2. 创建折线图显示月度趋势
    3. 创建柱状图显示产品类别对比
    4. 保存图表为PNG文件
    """
}

# 终止条件配置
TERMINATION_CONFIG = {
    "keywords": [
        "测试通过", "task completed", "任务完成", "successfully tested",
        "task completed successfully", "任务圆满完成", "all tests passed",
        "代码审查通过且测试完成", "final output ready"
    ],
    "min_messages": 3
}