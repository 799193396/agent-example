from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen-max-latest"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# LlamaIndex默认使用的大模型被替换为百炼
Settings.llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
# 加载本地的嵌入模型
Settings.embed_model = HuggingFaceEmbedding(model_name="D:\\llm\\Local_model\\BAAI\\bge-large-zh-v1___5")

text_to_sql_prompt_tmpl_str = """\
你是一个SQL专家。给定一个自然语言查询，您的工作是将其转换为SQL查询。
下面是一些如何将自然语言转换为SQL的例子：

<examples>
{{ examples }}
</examples>

现在轮到你了.
查询: {{ query_str }}
SQL: 
"""
# 添加几个案例
example_nodes = [
    TextNode(
        text="Query: llama2有多少参数?\nSQL: SELECT COUNT(*) FROM llama_2_params;"
    ),
    TextNode(
        text="Query: llama2有多少层?\nSQL: SELECT COUNT(*) FROM llama_2_layers;"
    ),
]
# 创建索引
index = VectorStoreIndex(nodes=example_nodes)

# 创建检索器
retriever = index.as_retriever(similarity_top_k=1)


def get_examples_fn(**kwargs):
    query = kwargs["query_str"]
    examples = retriever.retrieve(query)
    return "\n\n".join(node.text for node in examples)


# 使用函数映射到提示模板中，会使用检索器找寻对应的样例填充到提示模板里的examples中
prompt_tmpl = RichPromptTemplate(
    text_to_sql_prompt_tmpl_str,
    function_mappings={"examples": get_examples_fn},
)
# 组装问题到提示词中
prompt = prompt_tmpl.format(
    query_str="llama2模型的参数有多少?"
)
print(prompt)
# 使用大模型进行回答
response = Settings.llm.complete(prompt)
print(response.text)