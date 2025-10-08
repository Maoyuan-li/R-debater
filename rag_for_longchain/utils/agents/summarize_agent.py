# -*- coding: utf-8 -*-
# @Time    : 2025/05/xx
# @Author  : (patched by ChatGPT)
# @File    : summarize_agent.py
# @Desc    : ViewpointSummaryAgent（方法A补丁版：_robust_split 为实例方法，run 使用 self._robust_split）

from typing import List, Tuple, Optional
import re

def _normalize_str(x) -> str:
    return x if isinstance(x, str) else ""

class ViewpointSummaryAgent:
    """
    说明：
    - 保持原有对外接口：ViewpointSummaryAgent().run(user_input_chunks, generated_text) -> (my_adv, opp_adv, core_dis)
    - 仅做【方法A】相关改动：
        1) _robust_split 改为【实例方法】，不再使用 @staticmethod；
        2) run() 内调用 self._robust_split(summary)；
        3) 切分逻辑更宽松，兼容多种小标题写法与中英文冒号；
    - summarize_debate() 默认直接对 generated_text 做最小清洗返回（不改你原流程的话，此处可接入你已有的 LLM Summarize 调用）。
      若你已有外部模型摘要逻辑，可把调用放进 summarize_debate() 里。
    """

    def __init__(self, summarizer: Optional[callable] = None):
        """
        :param summarizer: 可选自定义摘要函数，形如 summarizer(text:str)->str。
                           如果不提供，则默认直接返回输入（最小行为，不影响你原有链路的稳定性）。
        """
        self._summarizer = summarizer

    # -------------------------
    # 你原来的摘要逻辑可以塞这里
    # -------------------------
    def summarize_debate(self, debate_text: str) -> str:
        """
        根据输入的辩论文本做一个结构化（或简单）摘要。这里保守处理：
        - 若你已有模型摘要逻辑，请在此处接入；
        - 若没有，就直接返回原文（再交给 _robust_split 做宽松切分/兜底）。
        """
        text = _normalize_str(debate_text).strip()
        if not text:
            return ""

        if self._summarizer is not None:
            try:
                out = self._summarizer(text)
                return _normalize_str(out).strip()
            except Exception:
                # 外部摘要失败就回退原文
                return text

        # 默认：不做额外处理，直接返回
        return text

    # -------------------------
    # 方法A：实例方法 + 宽松切分
    # -------------------------
    def _robust_split(self, summary: str) -> Tuple[str, str, str]:
        """
        宽松切分：兼容多种标题写法，避免模型微改导致切分失败。
        返回：(my_advantage, opponent_advantage, core_dispute)
        """
        s = _normalize_str(summary).strip()
        if not s:
            return "", "", ""

        # 兼容多种写法：占优/优势/要点；核心分歧/分歧/核心争议；中英文冒号/空格
        # 注意：正则只捕获标题标签，真正内容在标题之后直到下一个标题或文本末尾
        pos_pat = r"(正方(?:占优|优势|要点)\s*[：:])"
        neg_pat = r"(反方(?:占优|优势|要点)\s*[：:])"
        core_pat = r"((?:双\s*方)?(?:核\s*心)?(?:分\s*歧|争\s*议)(?:点)?\s*[：:])"

        matches = []
        for name, pat in (("pos", pos_pat), ("neg", neg_pat), ("core", core_pat)):
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                matches.append((name, m.start(), m.end()))

        if matches:
            # 按出现位置排序，逐段切
            matches.sort(key=lambda x: x[1])
            segs = []
            for i, (name, st, ed) in enumerate(matches):
                nxt = matches[i + 1][1] if i + 1 < len(matches) else len(s)
                segs.append((name, s[ed:nxt].strip()))
            pos = next((t for n, t in segs if n == "pos"), "")
            neg = next((t for n, t in segs if n == "neg"), "")
            core = next((t for n, t in segs if n == "core"), "")
            return pos, neg, core

        # 后备：按空行粗分，尽量给出三段
        blocks = [p.strip() for p in re.split(r"\n{2,}", s) if p.strip()]
        a = blocks[0] if blocks else ""
        b = blocks[1] if len(blocks) > 1 else ""
        c = blocks[2] if len(blocks) > 2 else ""
        return a, b, c

    # -------------------------
    # 对外主入口：与原有调用保持一致
    # -------------------------
    def run(self, user_input_chunks: List[str], generated_text: str) -> Tuple[str, str, str]:
        """
        :param user_input_chunks: 你的上游传入（通常是用户发言切片）；这里不强依赖，只为你需要时保留
        :param generated_text:   需要摘要/切分的文本（通常是上一阶段生成的辩论陈词）
        :return: (my_advantage, opponent_advantage, core_dispute)
        """
        # 这里仅用 generated_text 做摘要；若你要加“少量上下文”，可以拼接 user_input_chunks[:N]
        base_text = _normalize_str(generated_text)
        summary = self.summarize_debate(base_text)

        # 关键：方法A 使用实例方法调用
        my_adv, opp_adv, core_dis = self._robust_split(summary)

        # 统一做个轻微清洗（去掉多余空白）
        my_adv = my_adv.strip()
        opp_adv = opp_adv.strip()
        core_dis = core_dis.strip()
        return my_adv, opp_adv, core_dis



# 示例调用
if __name__ == "__main__":
    all_chunks = [
        "支持: 观点1的支持理由描述...",
        "支持: 观点2的支持理由描述...",
        "反对: 观点1的反对理由描述...",
        "反对: 观点2的反对理由描述..."
    ]
    counterargument = "支持: 额外的支持论点... 反对: 额外的反对论点..."

    agent = ViewpointSummaryAgent()
    pos, neg, core = agent.run(all_chunks, counterargument)
    print("正方占优论点：", pos)
    print("反方占优论点：", neg)
    print("核心分歧点：", core)
