# -*- coding: utf-8 -*-
# @Time    : 2025/3/27 18:48
# @Author  : Maoyuan Li
# @File    : EnhancedDebateAgent.py
# @Software: PyCharm
import numpy as np
import requests
import json
from typing import Dict, List, Callable, Any

import torch

from rag_for_longchain.generator.testgen import generate_counterargument_via_api

class DebateAgent:
    def __init__(self, stance: str, mode: str, api_url: str = None, api_key: str = None, custom_rag: Callable = None):
        assert stance in ["pro", "con", "mixed"], "stance 必须是 'pro' 或 'con' 或 'mixed'"
        self.stance = stance
        self.mode = mode
        self.api_url = api_url
        self.api_key = api_key
        self.custom_rag = custom_rag

    def generate_counterargument(
        self,
        debate_history: List[Dict[str, Any]],
        input_embedding,
        all_chunks,
        technique_details
    ) -> str:
        from rag_for_longchain.retriever.faiss_retriever import retrieve_candidates
        from rag_for_longchain.generator.testgen import process_chunks
        from rag_for_longchain.utils.agents.summarize_agent import ViewpointSummaryAgent
        from rag_for_longchain.utils.agents.debate_agent_new import main as final_main

        formatted_history = "\n".join([
            f"{turn['stance']} - {turn['debater']}: {turn['utterance']}"
            if turn['stance'] != 'mixed' else
            f"MIXED - ({turn['debater'].split('C')[0]} 质询 {turn['debater'].split('C')[1]}): {turn['utterance']}"
            for turn in debate_history
        ])

        if self.mode == "rag" and self.custom_rag:
            chunks, _ = retrieve_candidates(input_embedding, top_k=5)
            #print("debug",'chunks',chunks,'typechunks',type(chunks))
            result = process_chunks(chunks)
            all_texts_variable = "\n\n".join(result['所有文本'])

            counterargument = self.custom_rag(
                all_chunks=formatted_history,
                all_texts_variable=all_texts_variable,
                result={"最佳辩论技巧": result['最佳辩论技巧']}
            )

            summary_agent = ViewpointSummaryAgent()
            my_advantage, opponent_advantage, core_disagreement = summary_agent.run(all_chunks, counterargument)

            technique_detail = technique_details.get(result['最佳辩论技巧'], "")

            final_text, final_report = final_main(counterargument, my_advantage, opponent_advantage, core_disagreement, technique_detail)
            return final_text

        elif self.mode == "llm" and self.api_url and self.api_key:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            prompt = (
                f"以下是之前的辩论记录：\n{formatted_history}\n"
                f"你是 {'正方' if self.stance == 'pro' else '反方'}，请以清晰明确的立场进行强有力的回怼。"
            )
            data = {
                "model": "gpt-3.5-turbo",
                "stream": False,
                "messages": [
                    {"role": "system", "content": "你是一位逻辑清晰、针锋相对的辩手，请以明确立场回怼对手的观点，不要模棱两可。"},
                    {"role": "user", "content": prompt}
                ]
            }

            try:
                response = requests.post(f"{self.api_url}/chat/completions", headers=headers, json=data)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                return f"调用 LLM API 时发生错误：{e}"

        else:
            return "Agent 配置错误，无法生成回怼。"

class PKManager:
    def __init__(self, rag_agent: DebateAgent, llm_agent: DebateAgent, max_rounds: int = 3):
        self.rag_agent = rag_agent
        self.llm_agent = llm_agent
        self.max_rounds = max_rounds

    def start_debate(self, initial_debate: Dict[str, Any], input_embedding, all_chunks, technique_details):
        print("==== PK 结果 ====")
        print(f"比赛: {initial_debate['competition']}")
        print(f"对阵: {initial_debate['match']}")
        print(f"辩题: {initial_debate['topic']}")
        print(f"正方立场: {initial_debate['positions']['PRO']}")
        print(f"反方立场: {initial_debate['positions']['CON']}\n")

        debate_history = initial_debate['debate'].copy()
        for turn in debate_history:
            print(f"【{turn['stance']} - {turn['debater']}】\n{turn['utterance']}\n")

        for round_num in range(1, self.max_rounds + 1):
            agent = self.rag_agent if round_num % 2 == 1 else self.llm_agent
            response = agent.generate_counterargument(debate_history, input_embedding, all_chunks, technique_details)
            debate_turn = {
                "stance": agent.stance,
                "debater": f"{agent.mode.upper()}_R{round_num}",
                "utterance": response
            }
            debate_history.append(debate_turn)
            print(f"【{agent.mode.upper()} Agent 第 {round_num} 轮】\n{response}\n")

def generate_embeddings(chunks, tokenizer, model):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用最后一层隐藏状态的均值作为嵌入
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

def main():

    from transformers import AutoTokenizer, AutoModel
    from rag_for_longchain.generator.testgen import generate_counterargument_via_api

    model_name = "hfl/chinese-roberta-wwm-ext-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 构造 mock 数据
    initial_debate = {
        "competition": "测试赛",
        "match": "测试队A vs 测试队B",
        "topic": "是否应全面推行远程办公",
        "positions": {
            "PRO": "应全面推行远程办公",
            "CON": "不应全面推行远程办公"
        },
        "debate": [
            {"stance": "PRO", "debater": "P1", "utterance": "远程办公提高了员工的灵活性和工作满意度。"},
            {"stance": "CON", "debater": "C1", "utterance": "远程办公降低了团队协作效率，影响沟通。"}
        ]
    }

    all_chunks = ['{ "competition": "2014国际华语辩论邀请赛", "match": "复赛第二场 香港中文大学 vs 中山大学", "topic": "当今中国大陆大麻交易应/不应合法化", "positions": { "pro": "当今中国大陆大麻交易应该合法化", "con": "当今中国大陆大麻交易不应合法化" }, "debate": [ { "stance": "pro", "debater": "p1", "utterance": "我方主张大陆大麻交易合法化，通过法律明确低毒工业大麻与高毒医用大麻的区别，促进经济发展与有效监管。" }, { "stance": "mixed", "debater": "p1c4", "utterance": "辩友指出，大麻不仅在浓度上有别，还因用途不同需区分管理，合法化不等同于全民自由交易。" }, { "stance": "con", "debater": "c1", "utterance": "我方认为大陆吸食人数极低，合法化会增加执法成本和社会风险，加之历史原因，目前无变革必要。" }, { "stance": "mixed", "debater": "c1p4", "utterance": "辩友认为现行法律仅按数量定罪而未区分毒性，同时传统种植问题需要更明确的法规调整。" }, { "stance": "pro", "debater": "p2", "utterance": "我方总结合法化应有条件实施，既满足医疗与工业需求，又通过全国统一法规解决当前监管漏洞。" } ] }']

    input_embedding = generate_embeddings(all_chunks, tokenizer, model)

    technique_details = {
        "全面论证法": {
            "定义": "通过从多个不同角度进行论证，增加论据的全面性和说服力。",
            "场景背景": "在关于远程办公的辩论中，辩手从效率、幸福感、成本多个角度进行论证。",
            "示例文本": "“从效率看，远程办公减少了干扰；从幸福感看，灵活安排工作有利身心健康。”",
            "分析过程": "辩手多角度地支撑了观点。",
            "输出": "辩论技巧：全面论证法\n理由：使论点更丰富，有说服力。"
        }
    }

    rag_agent = DebateAgent(
        stance="pro",
        mode="rag",
        custom_rag=generate_counterargument_via_api
    )

    llm_agent = DebateAgent(
        stance="con",
        mode="llm",
        api_url="https://api.xi-ai.cn/v1",
        api_key="sk-qqNq0PZKCOyNvayA05378e4344Ef435dBd8d6d211aBaBc34"
    )

    manager = PKManager(rag_agent, llm_agent, max_rounds=2)
    manager.start_debate(initial_debate, input_embedding, all_chunks, technique_details)

if __name__ == "__main__":
    main()
