# -*- coding: utf-8 -*-
# @Time    : 2025/4/30 20:56
# @Author  : Maoyuan Li
# @File    : llm.py
# @Software: PyCharm
import os
import json
import re

import yaml
import requests

# —— 配置 —— #
CONFIG_FILE = "D:/converstional_rag/rag_for_longchain/config/config.yaml"
DEBATE_FOLDER = "D:\converstional_rag/23acldata\input_data\processed_input"  # 主文件夹路径
OUTPUT_FILE = "D:\converstional_rag/23acldata\output_data\LLM实验\debate_speeches_output_gpt4o.json"

def load_config(path: str) -> dict:
    """加载 OpenAI API 配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def call_llm(prompt: str, api_base: str, api_key: str, model: str) -> str:
    """向 LLM 发起请求，返回模型回复文本"""
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一名专业的辩论文本生成助手。"},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def generate_speeches_for_debate(debate_data: dict, api_base: str, api_key: str, model: str) -> dict:
    """根据辩论 JSON，调用 LLM 生成标题、正方/反方陈词"""
    debate_json = json.dumps(debate_data, ensure_ascii=False, indent=2)
    prompt = f"""
    以下给你一场辩论的元数据（JSON 格式）：
    {debate_json}
    请你产生输出，严格按照下面格式：
    辩论标题（Debate Title）
    使用的 LLM 模型名称（Model Used）
    正方陈词（Pro Speech），一段话
    反方陈词（Con Speech），一段话
    每一方陈词的字数不超过1100字
    用 JSON 对象返回，键分别是 "title"、"model"、"pro_speech"、"con_speech"。"""
    raw = call_llm(prompt, api_base, api_key, model)
    match = re.search(r'\{.*\}', raw, flags=re.S)
    json_str = match.group(0) if match else raw

    try:
        # 原有逻辑：调用 LLM，提取 JSON 到 result
        result = json.loads(json_str)

        # 强制覆盖 model 字段，确保和实际调用一样
        result["model"] = model

        return result
    except json.JSONDecodeError:
        # 如果解析依然失败，就把整个输出放到 pro_speech，确保 con_speech 不丢失
        return {
            "title": debate_data.get("match", "Unknown Match"),
            "model": model,
            "pro_speech": json_str,
            "con_speech": ""}

def main():
    cfg = load_config(CONFIG_FILE)
    api_conf = cfg.get("openai", {})
    api_base = api_conf.get("api_base")
    api_key = api_conf.get("api_key")
    model = api_conf.get("llm_model")

    outputs = []
    for sub in os.listdir(DEBATE_FOLDER):
        folder = os.path.join(DEBATE_FOLDER, sub)
        if not os.path.isdir(folder):
            continue

        # 找到辩论 JSON（非 last_two.json）
        debate_file = next(
            (f for f in os.listdir(folder)
             if f.endswith(".json") and f != "last_two.json"),
            None
        )
        if not debate_file:
            print(f"[跳过] {sub} 未找到辩论 JSON")
            continue

        # 使用文件名（去掉 .json）作为标准标题，确保高度一致
        debate_name = os.path.splitext(debate_file)[0]

        with open(os.path.join(folder, debate_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            result = generate_speeches_for_debate(data, api_base, api_key, model)
            # 强制覆盖 title 字段
            result["title"] = debate_name
            outputs.append(result)
            print(f"[完成] {debate_name}")
        except Exception as e:
            print(f"[错误] 处理 {debate_name} 时出错：{e}")

    # 写入输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"所有结果已保存到 {OUTPUT_FILE}")

if __name__ == '__main__':
    main()