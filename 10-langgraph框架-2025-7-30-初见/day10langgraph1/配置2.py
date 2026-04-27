from langgraph.graph import MessagesState
from langgraph.runtime import Runtime
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict


class MyContext(TypedDict):
    model: str


MODELS = {
    "anthropic": "anthropic:claude-3-5-haiku-latest",
    "openai": "openai:gpt-4.1-mini",
}


def call_model(state: MessagesState, runtime: Runtime[MyContext]):
    model = ""
    if runtime.context:
        model = runtime.context["model"]
        model = MODELS[model]
    return {"messages": model}


builder = StateGraph(MessagesState, context_schema=MyContext)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

# Usage
input_message = {"role": "user", "content": "hi"}
# With no configuration, uses default (Anthropic)
response_1 = graph.invoke({"messages": [input_message]})
# Or, can set OpenAI
context = {"model": "openai"}
response_2 = graph.invoke({"messages": [input_message]}, context=context)

print(response_1)
print(response_2)
