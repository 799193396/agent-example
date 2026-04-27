from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langchain_community.chat_models import ChatZhipuAI
from typing import TypedDict

# 定义状态结构
class MyState(TypedDict):
    question: str
    answer: str

# 定义配置结构
class MyContext(TypedDict):
    language: str  # 配置中包含语言选项，比如 "en" 或 "zh"

# 节点函数可以访问 config 参数
def step1(state: MyState, runtime: Runtime[MyContext]):
    if runtime.context["language"] == "zh":
        answer = "你好！"
    else:
        answer = "Hello!"
    return {"answer": answer}

# 构建图
graph = StateGraph(state_schema=MyState, context_schema=MyContext)
graph.add_node("step1", step1)
graph.set_entry_point("step1")

# 编译
app = graph.compile()

# 执行时传入 config 参数（区分于 state）
result = app.invoke({"question": "Hi"}, context={"language": "zh"})
print(result)  # => {"question": "Hi", "answer": "你好！"}