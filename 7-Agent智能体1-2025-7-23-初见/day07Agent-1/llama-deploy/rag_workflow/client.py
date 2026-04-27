import requests
import json

def call_workflow_http():
    # 调用部署的工作流
    url = "http://localhost:4501/deployments/rag_workflow/tasks/run"
    # 将参数编码为JSON字符串
    input_data = {
        "file_path": r"D:\\llm\\LLMProject\\LlamaIndex\\data\\小说.txt",
        "query": "萧炎的爸爸是谁"
    }

    payload = {
        "input": json.dumps(input_data, ensure_ascii=False)
    }

    response = requests.post(url, json=payload)
    result = response.json()
    print(result)


if __name__ == '__main__':
    call_workflow_http()