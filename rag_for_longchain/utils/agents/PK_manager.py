# -*- coding: utf-8 -*-
# @Time    : 2025/3/25 14:47
# @Author  : Maoyuan Li
# @File    : PK_manager.py
# @Software: PyCharm
from typing import Dict, Any

from rag_for_longchain.utils.agents.debate_agent import DebateAgent


class PKManager:
    """
    控制多轮 PK 过程
    """
    def __init__(self, rag_agent: DebateAgent, llm_agent: DebateAgent, max_rounds: int = 3):
        """
        初始化 PK 系统

        :param rag_agent: RAG Agent（使用 generate_counterargument_via_api）
        :param llm_agent: LLM-based Agent
        :param max_rounds: 最大回合数
        """
        self.rag_agent = rag_agent
        self.llm_agent = llm_agent
        self.max_rounds = max_rounds
        self.final_output = []  # 存储整个辩论过程的输出

    def start_debate(self, initial_debate: Dict[str, Any]):
        """
        开始多轮对战

        :param initial_debate: 初始辩论历史记录
        """
        print("==== PK 结果 ====")
        print(f"比赛: {initial_debate['competition']}")
        print(f"对阵: {initial_debate['match']}")
        print(f"辩题: {initial_debate['topic']}")
        print(f"正方立场: {initial_debate['positions']['PRO']}")
        print(f"反方立场: {initial_debate['positions']['CON']}\n")

        debate_history = initial_debate['debate'].copy()
        for turn in debate_history:
            output_text = f"【{turn['stance']} - {turn['debater']}】\n{turn['utterance']}\n"
            print(output_text)
            self.final_output.append(output_text)

        for round_num in range(1, self.max_rounds + 1):
            if round_num % 2 == 1:
                # RAG Agent 回怼
                response = self.rag_agent.generate_counterargument(debate_history)
                debate_turn = {"stance": self.rag_agent.stance, "debater": f"RAG_R{round_num}", "utterance": response}
                debate_history.append(debate_turn)
                output_text = f"【RAG Agent 第 {round_num} 轮】\n{response}\n"
                print(output_text)
                self.final_output.append(output_text)
            else:
                # LLM-based Agent 回怼
                response = self.llm_agent.generate_counterargument(debate_history)
                debate_turn = {"stance": self.llm_agent.stance, "debater": f"LLM_R{round_num}" if self.llm_agent.stance != 'mixed' else f"MIXED_R{round_num}", "utterance": response}
                debate_history.append(debate_turn)
                output_text = f"【LLM Agent 第 {round_num} 轮】\n{response}\n"
                print(output_text)
                self.final_output.append(output_text)

        # 翻译整个辩论过程
        translated_output = self.translate_to_english("\n".join(self.final_output))
        print("\n==== Debate Translation (English) ====\n")
        print(translated_output)

    def translate_to_english(self, text: str) -> str:
        """
        调用 LLM 进行翻译

        :param text: 需要翻译的文本
        :return: 翻译后的英文文本
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_agent.api_key}"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "messages": [
                {"role": "system", "content": "你是一个专业的翻译员，擅长将中文翻译成流畅的英语。"},
                {"role": "user", "content": f"请将以下中文文本翻译成流畅的英文：\n{text}"}
            ]
        }
        try:
            response = requests.post(f"{self.llm_agent.api_url}/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"翻译失败：{e}"