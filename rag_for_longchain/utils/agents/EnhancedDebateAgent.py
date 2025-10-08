# -*- coding: utf-8 -*-
# @Time    : 2025/3/27 18:48
# @Author  : Maoyuan Li
# @File    : EnhancedDebateAgent.py
# @Software: PyCharm
from typing import Callable, List, Any, Dict

from rag_for_longchain.utils.agents.debate_agent_new import DebateAgent


class EnhancedDebateAgent(DebateAgent):
    def __init__(self, stance: str, mode: str, api_url: str = None, api_key: str = None, custom_rag: Callable = None):
        super().__init__(stance, mode, api_url, api_key, custom_rag)
        self._first_inquiry = True
        self._judge_counter = 0
        self.judgment_debug_info = []

    def generate_and_validate_counterargument(self, debate_history: List[Dict[str, Any]],
                                              best_text: Any,
                                              process_chunks: Callable[[Any], Dict[str, Any]]) -> str:
        """
        生成回怼并验证其是否基于己方论点或针对对方论点进行反驳；
        如果不符合要求，则将原始提示扩充后重新生成。
        """
        # 生成初始回怼
        initial_response = self.generate_counterargument(debate_history, best_text, process_chunks)
        our_args, opponent_args = self.summarize_debate(debate_history)
        # 判断是否满足要求
        if self.judge_alignment(initial_response, our_args, opponent_args):
            return initial_response
        else:
            # 重新构造提示：包含原始提示和增强的双方观点
            original_prompt = self.construct_formatted_history(debate_history)
            refined_prompt = (
                    "【原始提示】：\n" + original_prompt + "\n\n" +
                    "当前生成的回怼文本未能充分基于双方论点进行说服或反驳，请注意：\n\n" +
                    "【我方论点总结】：\n" + our_args + "\n\n" +
                    "【对方论点总结】：\n" + opponent_args + "\n\n" +
                    "请基于上述原始提示和双方论点，重新生成你的回怼文本："
            )
            # 输出 refined_prompt 进行调试
            print("【Debug】 refined_prompt:")
            print(refined_prompt)
            print("【Debug】 refined_prompt输出完毕")

            # 将扩充后的提示传递给生成器函数
            refined_response = self.custom_rag(
                all_chunks=refined_prompt,
                all_texts_variable=refined_prompt,
                result={"生成文本": initial_response}
            )
            return refined_response

    def summarize_debate(self, debate_history: List[Dict[str, Any]]) -> (str, str):
        """
        将辩论历史按立场分类，汇总己方与对方论点。假设 'PRO' 与 'MIXED' 为己方，'CON' 为对方。
        """
        our_turns = [turn for turn in debate_history if turn['stance'].lower() in ['pro', 'mixed']]
        opponent_turns = [turn for turn in debate_history if turn['stance'].lower() == 'con']
        our_args = "\n".join([f"{turn['debater']}：{turn['utterance']}" for turn in our_turns])
        opponent_args = "\n".join([f"{turn['debater']}：{turn['utterance']}" for turn in opponent_turns])
        return our_args, opponent_args

    def judge_alignment(self, generated_text: str, our_args: str, opponent_args: str) -> bool:
        """
        使用 LLM 判断生成的文本是否基于己方论点生成或针对对方论点进行反驳。
        第一次判断时强制返回否以触发重新生成，并输出调试信息。
        """
        self._judge_counter += 1
        prompt = (
            "请判断下列生成文本是否是基于我方的论点进行生成，或者是针对对方论点进行反驳的？\n\n"
            "【生成文本】：\n" + generated_text + "\n\n" +
            "【我方论点】：\n" + our_args + "\n\n" +
            "【对方论点】：\n" + opponent_args + "\n\n"
            "请仅输出 '是' 或 '否' 表示是否符合要求，并附上简要理由。"
        )

        if self._first_inquiry:
            self._first_inquiry = False
            judgment_result = "否"
            judgment_explanation = "第一次判断时，强制返回否以触发重新生成。"
        else:
            llm_response = self.call_llm(prompt)
            if "是" in llm_response:
                judgment_result = "是"
            else:
                judgment_result = "否"
            judgment_explanation = llm_response

        debug_entry = {
            "计数器": self._judge_counter,
            "判别内容": prompt,
            "判别结果": judgment_result,
            "判别论述": judgment_explanation,
            "己方观点": our_args,
            "对方观点": opponent_args
        }
        self.judgment_debug_info.append(debug_entry)
        print("【判断调试信息】")
        for key, value in debug_entry.items():
            print(f"{key}: {value}")
        print("------")

        return judgment_result == "是"

    def call_llm(self, prompt: str) -> str:
        """
        调用 LLM 接口。这里依然使用 custom_rag 作为示例，你可以根据实际情况替换调用方式。
        """
        dummy_history = []
        response = self.custom_rag(all_chunks=dummy_history, all_texts_variable=prompt, result={})
        return response
