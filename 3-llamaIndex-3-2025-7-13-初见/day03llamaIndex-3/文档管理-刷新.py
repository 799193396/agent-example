from llama_index.core import SummaryIndex, Document

index = SummaryIndex([])
text_chunks = ["文档1", "文档2", "文档3"]

doc_chunks = []
for i, text in enumerate(text_chunks):
    doc = Document(text=text, id_=f"doc_id_{i}")
    doc_chunks.append(doc)

# 插入
for doc_chunk in doc_chunks:
    index.insert(doc_chunk)

print("更新前", index.docstore.docs)

# 刷新
# 修改第一个文档的内容
doc_chunks[0] = Document(text="全新的文档1内容", id_="doc_id_0")
# 新增一个新的文档
doc_chunks.append(Document(text="这是一个新增的文档哦", id_="doc_id_3"))

# 开始更新
ref_doc = index.refresh_ref_docs(doc_chunks)
print(ref_doc)
print("更新后", index.docstore.docs)