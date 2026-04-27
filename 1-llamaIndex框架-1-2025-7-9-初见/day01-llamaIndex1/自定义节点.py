from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

# 创建节点对象
node1 = TextNode(text="deepseek", id_="1")
node2 = TextNode(text="chatgpt", id_="2")
# 和其他节点进行关联，并添加元数据，对应关系：Node1对应{这是节点2}， node2对应{这是节点1}
node1.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
    node_id=node2.node_id, metadata={"这是节点1": "111"}
)
node2.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
    node_id=node1.node_id, metadata={"这是节点2": "222"}
)
nodes = [node1, node2]
print(nodes)