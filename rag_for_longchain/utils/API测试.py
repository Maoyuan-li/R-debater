import requests
import os
import json


def chat_completion(api_url, api_key):
    # 请求头设置
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 请求体数据
    data = {
        "model": "gpt-5-nano",
        "messages": [
            {
                "role": "developer",
                "content": "你是一个有帮助的助手。"
            },
            {
                "role": "user",
                "content": "你好！"
            }
        ]
    }

    try:
        # 发送POST请求
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # 检查请求是否成功

        # 解析并返回响应结果
        result = response.json()
        return result

    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        if response is not None:
            print(f"响应状态码: {response.status_code}")
            try:
                print(f"错误响应内容: {json.dumps(response.json(), indent=2)}")
            except:
                print(f"错误响应内容: {response.text}")
        return None


if __name__ == "__main__":
    # 从环境变量获取API地址和密钥，或者直接在这里设置
    api_url = os.getenv("NEWAPI_SERVER_URL", "https://www.xdaicn.top/v1/chat/completions")
    api_key = os.getenv("NEWAPI_API_KEY", "sk-VYRSPLZiXERVQkQRkU6EAP2xKneeCvUYOSMrrpVbyQMQ17lF")

    # 如果API密钥未设置，提示用户
    if not api_key:
        api_key = input("请输入你的NEWAPI_API_KEY: ")

    # 调用函数发送请求
    response = chat_completion(api_url, api_key)

    # 打印格式化的响应结果
    if response:
        print("\nAPI响应结果:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
