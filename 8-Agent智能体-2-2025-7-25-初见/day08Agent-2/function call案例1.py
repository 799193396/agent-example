import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from langchain.agents import create_openai_functions_agent
import os
import json
from io import StringIO
import numpy as np

# 加载环境变量
load_dotenv()

model = "qwen-max-0428"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# 初始化OpenAI客户端
client = OpenAI(api_key=api_key, base_url=api_base_url)

# 加载样例数据 - 更丰富的员工数据
df_employees = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Age': [25, 30, 35, 28, 32, 45],
    'Salary': [50000.0, 75000.5, 95000.75, 62000.0, 88000.25, 120000.0],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'Finance'],
    'IsMarried': [True, False, True, False, True, True],
    'YearsExperience': [3, 5, 8, 4, 7, 15]
})

# 将DataFrame转换为JSON格式,orient='split' 参数将数据、索引和列分开存储
df_json = df_employees.to_json(orient='split')
print("原始员工数据:")
print(df_employees)
print("\nJSON格式:", df_json)


"""
函数定义的规范
在使用Function Calling（函数调用）与聊天模型交互时，除了要规范输入数据格式外，函数编写也需要遵循以下核心规范：
1.语义清晰的命名规范
函数名应采用动宾结构，准确体现功能意图（如calculate_monthly_revenue）
避免使用模糊缩写，优先选择完整英文单词组合
2.结构化参数设计
参数排列遵循"核心参数优先"原则（如目标对象、主操作参数靠前）
命名采用snake_case格式，保持与函数名的语义连贯性（如start_date对应get_date_range）
3.自文档化描述体系
函数描述应包含三重说明：
▸ 核心功能（What）
▸ 参数约束（Why）
▸ 返回值规范（How）
每个参数需注明：
▸ 数据类型（type）
▸ 取值约束（constraints）

"""

# 函数1: 计算薪资统计信息
def calculate_salary_statistics(input_json):
    """
    计算薪资的统计信息：平均值、中位数、最高值、最低值

    参数:
    input_json (str): 包含员工数据的JSON格式字符串

    返回:
    str: 薪资统计信息，以JSON格式返回
    """
    try:
        # 重新将JSON数据恢复成DataFrame数据
        df = pd.read_json(StringIO(input_json), orient='split')
        salary_stats = {
            "average_salary": round(df['Salary'].mean(), 2),
            "median_salary": round(df['Salary'].median(), 2),
            "max_salary": round(df['Salary'].max(), 2),
            "min_salary": round(df['Salary'].min(), 2),
            "salary_range": round(df['Salary'].max() - df['Salary'].min(), 2)
        }
        return json.dumps(salary_stats)
    except Exception as e:
        return json.dumps({"error": str(e)})


# 函数2: 按部门分组统计
def analyze_by_department(input_json):
    """
    按部门统计员工数量、平均薪资和平均年龄

    参数:
    input_json (str): 包含员工数据的JSON格式字符串

    返回:
    str: 部门统计信息，以JSON格式返回
    """
    try:
        df = pd.read_json(StringIO(input_json), orient='split')
        # 根据部门进行分组，并开始计算分组后的数据
        dept_stats = df.groupby('Department').agg({
            'Name': 'count',
            'Salary': 'mean',
            'Age': 'mean'
        }).round(2)

        # 组转最终返回的内容
        result = {}
        for dept in dept_stats.index:
            result[dept] = {
                "employee_count": int(dept_stats.loc[dept, 'Name']),  #  获取上面分组的员工数量
                "average_salary": round(dept_stats.loc[dept, 'Salary'], 2),
                "average_age": round(dept_stats.loc[dept, 'Age'], 2)
            }

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# 函数3: 查找满足条件的员工
def find_employees_by_criteria(input_json, min_salary=None, max_age=None, department=None):
    """
    根据指定条件查找员工

    参数:
    input_json (str): 包含员工数据的JSON格式字符串
    min_salary (float): 最低薪资要求（可选）
    max_age (int): 最大年龄限制（可选）
    department (str): 指定部门（可选）

    返回:
    str: 符合条件的员工信息，以JSON格式返回
    """
    try:
        df = pd.read_json(StringIO(input_json), orient='split')
        filtered_df = df.copy()

        # 应用筛选条件
        if min_salary is not None:
            filtered_df = filtered_df[filtered_df['Salary'] >= min_salary]
        if max_age is not None:
            filtered_df = filtered_df[filtered_df['Age'] <= max_age]
        if department is not None:
            filtered_df = filtered_df[filtered_df['Department'] == department]

        # 转换为可序列化的格式
        result = {
            "matching_employees": filtered_df[['Name', 'Age', 'Salary', 'Department']].to_dict('records'),
            "count": len(filtered_df)
        }

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# 函数4: 计算经验与薪资相关性
def analyze_experience_salary_correlation(input_json):
    """
    分析工作经验与薪资的相关性

    参数:
    input_json (str): 包含员工数据的JSON格式字符串

    返回:
    str: 相关性分析结果，以JSON格式返回
    """
    try:
        df = pd.read_json(StringIO(input_json), orient='split')
        correlation = df['YearsExperience'].corr(df['Salary'])

        # 简单的线性回归分析，工作年限与工资的变化
        z = np.polyfit(df['YearsExperience'], df['Salary'], 1)

        result = {
            "correlation_coefficient": round(correlation, 4),  # 返回结果的第一项
            "correlation_strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(
                correlation) > 0.3 else "weak",  #  根据相关系数的绝对值大小，判断其强度
            "salary_increase_per_year": round(z[0], 2),  # 线性回归的斜率部分，表示 每年经验提升对应的薪资增加
            "base_salary_estimate": round(z[1], 2)  # 截距部分：当一个人刚入行时（0年经验），预测的起薪
        }

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# 定义函数库
# 函数库对象必须是一个字典，一个键值对代表一个函数，其中Key是代表函数名称的字符串，而value表示对应的函数。
function_repository = {
    "calculate_salary_statistics": calculate_salary_statistics,
    "analyze_by_department": analyze_by_department,
    "find_employees_by_criteria": find_employees_by_criteria,
    "analyze_experience_salary_correlation": analyze_experience_salary_correlation,
}

# 定义所有可用的工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_salary_statistics",
            "description": "计算员工薪资的统计信息，包括平均值、中位数、最高值、最低值等",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_json": {
                        "type": "string",
                        "description": "包含员工数据的JSON格式字符串"
                    }
                },
                "required": ["input_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_by_department",
            "description": "按部门分析员工数据，统计各部门的员工数量、平均薪资和平均年龄",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_json": {
                        "type": "string",
                        "description": "包含员工数据的JSON格式字符串"
                    }
                },
                "required": ["input_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_employees_by_criteria",
            "description": "根据指定的条件（最低薪资、最大年龄、部门）查找符合条件的员工",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_json": {
                        "type": "string",
                        "description": "包含员工数据的JSON格式字符串"
                    },
                    "min_salary": {
                        "type": "number",
                        "description": "最低薪资要求（可选）"
                    },
                    "max_age": {
                        "type": "integer",
                        "description": "最大年龄限制（可选）"
                    },
                    "department": {
                        "type": "string",
                        "description": "指定部门名称（可选）"
                    }
                },
                "required": ["input_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_experience_salary_correlation",
            "description": "分析工作经验与薪资之间的相关性，提供相关系数和趋势分析",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_json": {
                        "type": "string",
                        "description": "包含员工数据的JSON格式字符串"
                    }
                },
                "required": ["input_json"]
            }
        }
    }
]


def run_query(query):
    """运行单个查询"""
    print(f"\n{'=' * 60}")
    print(f"查询: {query}")
    print('=' * 60)

    # 构建messages
    messages = [
        {
            "role": "system",
            "content": f"你是一位专业的数据分析师。现在有一个员工数据集：{df_json}。请根据用户的要求选择合适的分析函数来处理数据。"
        },
        {
            "role": "user",
            "content": query
        }
    ]

    try:
        # 第一次API调用,是否决定调用工具，该使用哪个工具
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        # 检查是否有函数调用
        if response.choices[0].message.tool_calls:
            # 获取对应的函数用
            tool_call = response.choices[0].message.tool_calls[0]
            # 获取调用的函数名称
            function_name = tool_call.function.name
            # 获取调用的函数参数
            function_args = json.loads(tool_call.function.arguments)
            # 获取函数的id
            tool_call_id = tool_call.id

            print(f"调用函数: {function_name}")
            print(f"函数参数: {function_args}")

            # 执行本地函数
            local_function = function_repository[function_name]
            # 将参数传入函数中，并获取结果
            function_response = local_function(**function_args)

            print(f"函数执行结果: {function_response}")

            # 添加响应到messages
            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "name": function_name,
                "tool_call_id": tool_call_id,
                "content": function_response
            })

            # 第二次API调用获取最终答案
            final_response = client.chat.completions.create(
                model=model,
                messages=messages,
            )

            print(f"\n最终答案: {final_response.choices[0].message.content}")

        else:
            print(f"直接回答: {response.choices[0].message.content}")

    except Exception as e:
        print(f'查询出错: {e}')


# 运行示例查询
if __name__ == "__main__":
    print("开始运行数据分析查询示例...")
    # 测试不同的查询示例
    test_queries = [
        "请计算所有员工的薪资统计信息",
        "请分析各个部门的员工情况",
        "请找出薪资超过80000且年龄小于35岁的IT部门员工",
        "请分析工作经验与薪资的相关性"
    ]

    for query in test_queries:
        run_query(query)