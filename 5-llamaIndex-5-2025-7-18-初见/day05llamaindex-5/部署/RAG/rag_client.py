from llama_deploy import Client
import asyncio


async def run_rag_workflow():
    # 创建客户端
    client = Client(control_plane_url="http://localhost:8000")

    # 获取会话
    session = await client.core.sessions.create()

    # try:
    print("Loading documents...")
    query_result = await session.run(
        service_name="rag_workflow",
        # arg="你好"  # 电脑内存小于16G的执行一下代码
        arg={  # 电脑内存大于16G的执行一下代码
            "file_path": r"../../../data/小说.txt",
            "query": "萧炎的爸爸是谁？"
        }
    )
    print(f"Query result: {query_result}")



# 第三个启动的程序
if __name__ == "__main__":
    # 异步版本
    asyncio.run(run_rag_workflow())
