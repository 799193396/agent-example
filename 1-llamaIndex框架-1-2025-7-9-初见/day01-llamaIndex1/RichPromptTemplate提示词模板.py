from llama_index.core.prompts import RichPromptTemplate

context_str = """
    DeepSeek，全称杭州深度求索人工智能基础技术研究有限公司 [40]。DeepSeek是一家创新型科技公司 [3]，成立于2023年7月17日 [40]，使用数据蒸馏技术 [41]，得到更为精练、有用的数据 [41]。
    由知名私募巨头幻方量化孕育而生 [3]，专注于开发先进的大语言模型（LLM）和相关技术 [40]。注册地址 [6]：浙江省杭州市拱墅区环城北路169号汇金国际大厦西1幢1201室 [6]。法定代表人为裴湉 [6]，
    经营范围包括技术服务、技术开发、软件开发等 [6]。
"""
question = 'deepseek成立于哪一年？'
template = RichPromptTemplate(
    """我们在下面提供了上下文信息
    ---------------------
    {{ context_str }}
    ---------------------
    有了这些信息，请回答问题: {{ query_str }}
    """
)

# 格式化为字符串
prompt_str = template.format(context_str=context_str, query_str=question)
print(prompt_str)
# 格式化聊天消息列表
messages = template.format_messages(context_str=context_str, query_str=question)
print(messages)

from llama_index.core.prompts import RichPromptTemplate

template = RichPromptTemplate(
    """
{% chat role="system" %}
给定一个列表，包含图片和文本, 请尽你所能回答这个问题.
{% endchat %}

{% chat role="user" %}
{% for image_path, text in images_and_texts %}
这是一些文本: {{ text }}
这是一张图片:
{{ image_path | image }}
{% endfor %}
{% endchat %}
"""
)

messages = template.format_messages(
    images_and_texts=[
        ("page_1.jpg", "文件的第一页数据"),
        ("page_2.jpg", "文件的第二页数据"),
    ]
)
print(messages)