from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from util import load_model

# 加载大模型和嵌入模型
llm, embed_model = load_model.get_llm()

print("-----------基础使用-----------")
# chat_engine = SimpleChatEngine.from_defaults(llm=llm)
# 创建聊天引擎
# response = chat_engine.chat("我今天吃了火锅，心情很不错。")
# print(response)
#
# res = chat_engine.chat("我今天吃了什么？")
# print(res)

print("-----------使用索引构建-高级API------------")
# 加载文档
documents = SimpleDirectoryReader(input_files=["data/小说.txt"]).load_data()
splitter = SentenceSplitter(
    chunk_size=200,
    chunk_overlap=100,
    separator="-----",  # 拼接句子的分隔符
    paragraph_separator="\n\n"  # 拼接段落的分隔符

)
# 创建索引和检索器
index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
# # # 创建聊天引擎
# chat_engine = index.as_chat_engine(similarity_top_k=10, chat_mode="condense_plus_context", verbose=True)
# print(chat_engine.chat("萧炎斗之力是多少段？"))
# 聊天记录会自动报错到内存中
# 第二次对话 重写问题：萧薰儿的斗之力是九段。而萧炎的斗之力是三段。因此，萧薰儿的斗之力比萧炎高六段。
# print(chat_engine.chat("萧薰儿的斗之力是多少？比萧炎高多少？"))


print("==============低级API-手动构造Chat Engine,能够达到更精细的定制=================")
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine

custom_prompt = PromptTemplate(
    """\
    根据以下人类与助手之间的对话记录，以及人类提出的后续问题，\
    请将该后续问题改写为一个完整的、自包含的问题，使其能够在没有对话上下文的情况下也能被准确理解。

    <对话历史>
    {chat_history}

    <后续问题>
    {question}

    <完整问题>
    """
)

# 构建历史消息
custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="萧炎斗之力是多少段？",
    ),
    ChatMessage(role=MessageRole.ASSISTANT,
                content="根据文档中的信息，萧炎的斗之力是三段。这在第一章中明确提到：“斗之力，三段！”并且还描述了他在测验魔石碑上看到这个结果时的情景。"),
]
# 创建查询引擎
query_engine = index.as_query_engine(similarity_top_k=10)
# 手动创建聊天引擎（先从对应的历史消息中组合当前问题去生成一个新的问题，把新问题交给查询引擎去进行回复）
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=custom_prompt,
    chat_history=custom_chat_history,
    verbose=True,
)
# 普通输出
print(chat_engine.chat("萧薰儿的斗之力是多少？比萧炎高多少？"))

# 流式输出
streaming_response = chat_engine.stream_chat("萧薰儿的斗之力是多少？比萧炎高多少？")
for token in streaming_response.response_gen:
    print(token, end="")