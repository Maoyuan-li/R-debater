# rdebater_model.py
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional
import os
import json
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from rag_for_longchain.retriever.npz_keyword_retriever import search
from rag_for_longchain.generator.testgen_for_pk import (
    generate_counterargument_via_api,
    load_debate_techniques
)
from rag_for_longchain.utils.agents.summarize_agent import ViewpointSummaryAgent
from rag_for_longchain.utils.agents.debate_agent_new import main as debate_main


def _safe_load_model_local_first(model_name: str, local_dir: Optional[str] = None):
    """
    优先从本地目录加载（需要存在权重：model.safetensors 或 pytorch_model.bin）。
    若本地不存在权重，则尝试在线加载；都失败时抛出明确错误。
    """
    def _has_weights(d: Optional[str]) -> bool:
        if not d:
            return False
        return (
            os.path.exists(os.path.join(d, "model.safetensors")) or
            os.path.exists(os.path.join(d, "pytorch_model.bin"))
        )

    # —— 本地优先：完全离线 ——
    if _has_weights(local_dir):
        # 强制离线，避免任何到 huggingface.co 的探测请求
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        mdl = AutoModel.from_pretrained(local_dir,  local_files_only=True)
        # 打印来源（可注释）
        print(f"[RDebater] 模型本地加载：{local_dir}")
        return tok, mdl

    # —— 在线兜底（需要网络可用） ——
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        print(f"[RDebater] 模型在线加载：{model_name}")
        return tok, mdl
    except Exception as e:
        # 如仍提供了 local_dir（但没权重），再尝试“仅本地”（能更清晰报错）
        if local_dir and os.path.isdir(local_dir):
            try:
                tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
                mdl = AutoModel.from_pretrained(local_dir,  local_files_only=True)
                print(f"[RDebater] 模型本地加载（兜底）：{local_dir}")
                return tok, mdl
            except Exception:
                pass
        raise RuntimeError(
            f"[HF加载失败] 既无法下载 {model_name}，也无法从本地目录加载：{local_dir}\n"
            f"请确认本地目录含权重文件（pytorch_model.bin 或 model.safetensors）。\n"
            f"原始异常：{repr(e)}"
        ) from e


class RDebaterModel:
    DEFAULT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
    DEFAULT_TECH_FILE = "D:/converstional_rag/rag_for_longchain/utils/debate_techniques.json"

    def __init__(self, config_file: str):
        cfg = self._load_config(config_file)

        # safe get with default
        self.model_name = cfg.get("MODEL_NAME", self.DEFAULT_MODEL_NAME)
        self.techniques_file = cfg.get("DEBATE_TECHNIQUES_FILE", self.DEFAULT_TECH_FILE)

        # 本地目录（优先级：config.yaml > 环境变量RDEBATER_MODEL_LOCAL_DIR > 环境变量ENC_LOCAL_DIR_HFL > 默认值）
        self.local_dir = (
            cfg.get("MODEL_LOCAL_DIR")
            or os.getenv("RDEBATER_MODEL_LOCAL_DIR")
            or os.getenv("ENC_LOCAL_DIR_HFL")
            or r"D:\hf_models\hfl-chinese-roberta-wwm-ext-large"
        )

        # 关键：只加载一次！不要再覆盖 self.model
        self.tokenizer, self.model = _safe_load_model_local_first(
            self.model_name, local_dir=self.local_dir
        )

        # 加载辩论技巧库
        self.techniques = load_debate_techniques(self.techniques_file)

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise RuntimeError(f"加载配置失败：{e}")

    def generate_utterance(
        self,
        debate_json: dict,
        context: str,
        stance: str
    ) -> str:
        """
        1）保证 debate_json["positions"] 存在且格式正确；
        2）提取 history→检索→生成初稿→摘要→最终润色；
        返回本轮的 utterance。
        """
        # —— 保护：positions 字段 —— #
        positions = debate_json.get("positions") or {}
        positions.setdefault("PRO", "")
        positions.setdefault("CON", "")
        debate_json["positions"] = positions

        # 1) 提取历史 + 合并 context
        history = [e["utterance"].strip()
                   for e in debate_json.get("debate", [])
                   if isinstance(e.get("utterance"), str)]
        if isinstance(context, str) and context.strip():
            history.append(context.strip())

        # 2) 切分 + “向量”生成（本模型充当文本编码器） + 检索
        chunks = self._recursive_split("\n\n".join(history), max_length=512)
        embs = self._generate_embeddings(chunks)  # 如果后续没用到 embs，可按需移除
        best_chunks, _ = search(chunks)           # 你自己的检索函数
        proc = self._process_chunks(best_chunks)
        retrieved_text = "\n\n".join(proc["所有文本"])
        best_skill = proc["最佳辩论技巧"]

        # 3) 生成初稿
        init_draft = generate_counterargument_via_api(
            debate_json,
            retrieved_text,
            {"最佳辩论技巧": best_skill},
            stance=stance
        )

        # 4) 提炼优势 & 核心分歧
        my_adv, opp_adv, core_dis = ViewpointSummaryAgent().run(history, init_draft)

        # 5) 最终润色
        technique_detail = self.techniques.get(best_skill, {}) if best_skill else {}
        final_text, _ = debate_main(
            init_draft,
            my_adv,
            opp_adv,
            core_dis,
            technique_detail
        )
        return final_text

    # ===== 私有方法 =====
    def _recursive_split(self, text: str, max_length: int):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= max_length:
            return [text]
        mid = len(tokens) // 2
        part1 = self.tokenizer.convert_tokens_to_string(tokens[:mid])
        part2 = self.tokenizer.convert_tokens_to_string(tokens[mid:])
        return self._recursive_split(part1, max_length) + self._recursive_split(part2, max_length)

    def _generate_embeddings(self, chunks: list):
        embs = []
        for c in chunks:
            inputs = self.tokenizer(c, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = self.model(**inputs)
            vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embs.append(vec)
        return np.vstack(embs)

    def _process_chunks(self, chunks: list) -> dict:
        texts, scores = [], {}
        for c in chunks:
            texts.append(c.get("text", "") if isinstance(c, dict) else str(c))
            labels = c.get("labels", {}) if isinstance(c, dict) else {}
            for skill, info in labels.items():
                try:
                    sc = float(info.get("评分", 0))
                except Exception:
                    sc = 0.0
                scores[skill] = scores.get(skill, 0.0) + sc

        if scores:
            best = max(scores, key=scores.get)
            return {"所有文本": texts, "最佳辩论技巧": best, "最高评分": scores[best]}
        else:
            return {"所有文本": texts, "最佳辩论技巧": None, "最高评分": 0.0}


# 使用示例（建议把路径写成原始字符串 r"..." 以避免反斜杠转义）
if __name__ == "__main__":
    sample_path = r"D:\converstional_rag\23acldata\matches\1_4决赛_中山大学珠海校区VS上海对外经贸大学_中华文化浪漫在繁_简.json"
    cfg_path    = r"D:\converstional_rag\rag_for_longchain\config\config.yaml"

    # 可选：显式告诉模型本地目录
    os.environ.setdefault("RDEBATER_MODEL_LOCAL_DIR", r"D:\hf_models\hfl-chinese-roberta-wwm-ext-large")

    data = json.loads(Path(sample_path).read_text(encoding="utf-8"))
    rdebater = RDebaterModel(cfg_path)
    print("Pro:", rdebater.generate_utterance(data, context="", stance="pro"))
    print("Con:", rdebater.generate_utterance(data, context="", stance="con"))
