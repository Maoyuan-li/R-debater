#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import yaml
from openai import OpenAI
import re, json
class DebateChunkLabeler:
    FEW_SHOT_PATH = "D:/converstional_rag/rag_for_longchain/utils/debate_skill.txt"
    CONFIG_PATH   = "D:/converstional_rag/rag_for_longchain/config/config.yaml"
    MARKING_CRITERIA_PATH = "D:/converstional_rag/rag_for_longchain/utils/mark_debate_skill.txt"

    def __init__(self, chunks):
        """
        初始化，加载配置、示例、评分细则，并初始化 OpenAI 客户端。
        """
        self.chunks = chunks
        self._load_config()
        self._load_marking_criteria()
        with open(self.FEW_SHOT_PATH, 'r', encoding='utf-8') as f:
            self.few_shot_example = f.read()

        # 新接口客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,  # ← 把 api_base 改名为 base_url
        )

    def _load_config(self):
        """
        从 YAML 中读取 openai.api_key, api_base, llm_model
        """
        with open(self.CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f).get("openai", {})
        self.api_key  = cfg.get("api_key")
        self.api_base = cfg.get("api_base")
        self.model    = cfg.get("llm_model", "gpt-4")

    def _load_marking_criteria(self):
        """
        读取评分细则文本
        """
        with open(self.MARKING_CRITERIA_PATH, 'r', encoding='utf-8') as f:
            self.marking_criteria = f.read()

    def clean_and_validate_json(self, content: str):
        text = content.strip()

        # 去掉 markdown 围栏
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 抓取第一个 {...}
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match:
                return json.loads(match.group(0))
            raise ValueError("Invalid JSON format returned by the model.")

    def label_chunk(self, chunk: dict) -> dict:
        """
        调用 OpenAI 客户端对单个 chunk 进行标注，并打印原始输出，便于调试。
        """
        prompt = f"""
    你是一个辩论分析助手。根据以下论证范式的描述，为辩论发言标注所有包含的论证范式，并对每个论证范式在辩论过程中的运用进行评分。

    ### 论证范式示例
    {self.few_shot_example}

    ### 评分细则：
    {self.marking_criteria}

    ### 待标注的发言：
    {chunk['utterance']}

    请严格仅输出一个 JSON 对象，格式如下：
    {{
        "论证范式1": {{"理由": "理由1", "评分": 分数}},
        "论证范式2": {{"理由": "理由2", "评分": 分数}},
        ...
    }}

    如果未找到任何论证范式，请输出：{{}}。
    不要输出任何解释或代码块标记。
    """

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "你是一个辩论分析专家，专注于对辩论内容进行分类、技巧标注和评分。只输出 JSON 对象，不要附加说明。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
                # 如果你用 gpt-4o-mini，可以加上 response_format={"type": "json_object"}
            )

            # 原始模型输出
            text = resp.choices[0].message.content.strip()
            print("=== 原始模型输出 ===")
            print(text)
            print("==================")

            # 尝试解析
            try:
                chunk["labels"] = self.clean_and_validate_json(text)
            except Exception as e:
                chunk["labels"] = {
                    "Error": str(e),
                    "RawOutput": text  # 保存原始输出，方便后续分析
                }

        except Exception as e:
            chunk["labels"] = {"Error": f"调用接口失败: {e}"}

        return chunk

    def process_and_label(self):
        """
        批量对所有 chunks 打标签并评分。
        """
        results = []
        for c in self.chunks:
            print("标注中：", c["utterance"][:30], "…")
            results.append(self.label_chunk(c))
        return results

if __name__ == "__main__":
    # 示例 chunks
    chunks_data = [
        {"stance": "支持", "debater": "A", "utterance": "我认为这个政策能带来巨大的经济效益。"},
        {"stance": "反对", "debater": "B", "utterance": "但这样的政策忽视了环境保护的重要性。"}
    ]

    labeler = DebateChunkLabeler(chunks_data)
    labeled = labeler.process_and_label()
    print(json.dumps(labeled, ensure_ascii=False, indent=4))
