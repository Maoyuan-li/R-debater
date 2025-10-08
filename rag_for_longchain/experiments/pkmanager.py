# -*- coding: utf-8 -*-
# @Time    : 2025/7/4 19:54
# @Author  : Maoyuan Li
# @File    : pkmanager.py
# @Software: PyCharm
# pk_manager.py

import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

# —————————————— 1. 定义一个通用的 DebateModel 接口 ——————————————
class DebateModel(ABC):
    """
    抽象基类，所有辩论模型都要实现这个接口。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """模型名字，用于在 debate entry 里打标签"""
        pass

    @abstractmethod
    def generate_utterance(
        self,
        debate_json: Dict,
        context: Optional[str],
        stance: str
    ) -> str:
        """
        生成一次对立方（stance）的发言。
        - debate_json: 当前整个 JSON 数据（包含 history、topic、positions…）
        - context: 上一轮对手最新的发言，或检索到的背景文本
        - stance: 'pro' or 'con'
        返回：一段字符串 utterance
        """
        pass


# —————————————— 2. 给 R_debater 包装一个实现 ——————————————
class RDebaterModel(DebateModel):
    def __init__(self, config):
        # 这里把你 batch_debate_generate.py 里初始化的部分抽出来
        from transformers import AutoTokenizer, AutoModel
        import yaml

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])
        self.model = AutoModel.from_pretrained(config["MODEL_NAME"])
        self.DEBATE_TECHNIQUES = __import__("rag_for_longchain.generator.testgen")\
            .load_debate_techniques(config["DEBATE_TECHNIQUES_FILE"])

    @property
    def name(self):
        return "R_debater"

    def generate_utterance(self, debate_json, context, stance):
        # 1) 调用 generate_counterargument_via_api 产生初稿
        from rag_for_longchain.generator.testgen import generate_counterargument_via_api
        init = generate_counterargument_via_api(
            debate_json,
            context or "",
            {"最佳辩论技巧": debate_json.get("最佳辩论技巧", "")},
            stance=stance
        )
        # 2) 调用 ViewpointSummaryAgent 提取优势和核心分歧
        from rag_for_longchain.utils.agents.summarize_agent import ViewpointSummaryAgent
        my_adv, opp_adv, core_dis = ViewpointSummaryAgent().run(
            [utt["utterance"] for utt in debate_json.get("debate", [])], init
        )
        # 3) 调用 debate_main 完成最终输出
        from rag_for_longchain.utils.agents.debate_agent_new import main as debate_main
        final, _ = debate_main(init, my_adv, opp_adv, core_dis,
                               self.DEBATE_TECHNIQUES.get(debate_json.get("最佳辩论技巧", ""), {}))
        return final


# —————————————— 3. 其他模型的示例包装 ——————————————
class SimpleRAGModel(DebateModel):
    def __init__(self, rag_agent):
        self._rag_agent = rag_agent

    @property
    def name(self):
        return "Simple_RAG"

    def generate_utterance(self, debate_json, context, stance):
        # 假设 rag_agent 有个统一接口 run(data, context, stance)
        return self._rag_agent.run(debate_json, context, stance)


class LLMModel(DebateModel):
    def __init__(self, llm_client):
        self._llm = llm_client

    @property
    def name(self):
        return "OpenAI-Chat"

    def generate_utterance(self, debate_json, context, stance):
        prompt = f"""你是辩论裁判。当前立场：{stance}。
历史对话：
{json.dumps(debate_json.get("debate", []), ensure_ascii=False, indent=2)}
请给出 {stance} 观点的下一段发言。"""
        return self._llm.chat(prompt)


# ----------------—— 4. PK Manager 核心逻辑（替换此类） ——----------------
from typing import Dict

class PKManager:
    def __init__(self, pro_model: DebateModel, con_model: DebateModel, max_rounds: int = 3):
        """
        pro_model: 只用于正方发言的模型实例（例如 FixedStanceModel(rdebater, 'pro')）
        con_model: 只用于反方发言的模型实例（例如 FixedStanceModel(naive,   'con')）
        max_rounds: 正反交替的轮数（默认 3）
        """
        self.pro_model = pro_model
        self.con_model = con_model
        self.max_rounds = max_rounds

    @staticmethod
    def _last_utterance_str(debate_json: Dict) -> str:
        """
        取上一条发言文本，确保返回 str，避免 list.strip 报错。
        """
        debate = debate_json.get("debate", [])
        if not debate:
            return ""
        last_item = debate[-1]
        if isinstance(last_item, dict):
            utt = last_item.get("utterance", "")
            return utt if isinstance(utt, str) else ""
        return ""

    def run_debate(self, debate_json: Dict) -> Dict:
        """
        每一轮严格执行：PRO 回合 -> CON 回合
        共执行 max_rounds 轮；总计 append 2 * max_rounds 条发言。
        """
        debate_json.setdefault("debate", [])

        for _ in range(self.max_rounds):
            # ---- PRO 回合 ----
            context_pro = self._last_utterance_str(debate_json)
            pro_utt = self.pro_model.generate_utterance(debate_json, context_pro, stance="pro")
            debate_json["debate"].append({
                "stance": "PRO",
                "debater": getattr(self.pro_model, "name", type(self.pro_model).__name__),
                "utterance": pro_utt
            })

            # ---- CON 回合 ----
            context_con = self._last_utterance_str(debate_json)
            con_utt = self.con_model.generate_utterance(debate_json, context_con, stance="con")
            debate_json["debate"].append({
                "stance": "CON",
                "debater": getattr(self.con_model, "name", type(self.con_model).__name__),
                "utterance": con_utt
            })

        return debate_json



# —————————————— 5. 启动示例 ——————————————
if __name__ == "__main__":
    # 1) 读入你的 JSON
    path = Path("input/debate_example.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    # 2) 实例化各模型
    config = {
      "MODEL_NAME": "hfl/chinese-roberta-wwm-ext-large",
      "DEBATE_TECHNIQUES_FILE": "path/to/debate_techniques.json"
    }
    rdebater = RDebaterModel(config)
    simple_rag = SimpleRAGModel(rag_agent=YourRAGAgent(...))
    openai_llm = LLMModel(llm_client=YourLLMClient(api_key=...))

    # 3) 运行 PK
    manager = PKManager([rdebater, simple_rag, openai_llm], max_rounds=5)
    result = manager.run_debate(data)

    # 4) 存盘
    Path("output/debate_pk_result.json")\
      .write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PK 结束，结果写到 debate_pk_result.json")
