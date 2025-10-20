# -*- coding: utf-8 -*-
# npz_keyword_retriever.py
"""
NPZ 关键词检索（列表查询版，使用大语言模型抽取“面向辩论”的关键词）
- 输入：query_list（list[str]），每个元素是一段辩论陈词/问题
- 步骤：逐段用 LLM 抽“辩论检索关键词” -> 聚合打分 -> 选全局Top-5关键词 -> BM25检索Top-5文段
- 输出：results, chunk_scores
- LLM接口：OpenAI兼容 /v1/chat/completions（已写死 URL），请在 API_KEY 处填入你的密钥
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import time
import re
import numpy as np
import requests

# ========= 固定参数（按需改成你的真实路径）=========
NPZ_PATH = r"D:\conversational_rag\rag_for_longchain\data\all.npz"

# 每条子查询让 LLM 抽取的关键词数量（短语优先）
PER_ITEM_K = 8
# 合并后用于BM25检索的关键词数（需求：五个）
FINAL_TOP_KW = 5
# BM25召回的段落数（需求：topk=5）
BM25_TOPK = 5

# === LLM API（已写死） ===
API_BASE = "https://www.xdaicn.top/v1"
API_KEY = "sk-p94mUCY5g2jvxd4i5iAReGUI0rqwd4DDP55EOGenDV2VSDNm"   # <<< 在此替换成你的真实 API Key
MODEL_NAME = "gpt-4o-mini"
TIMEOUT_SEC = 30
MAX_RETRY = 2

# === 依赖（必须） ===
try:
    from rank_bm25 import BM25Okapi
    import jieba
    import jieba.analyse
except ImportError as e:
    raise ImportError("请先安装依赖：pip install rank-bm25 jieba") from e


# ========= 工具函数 =========
def _normalize_utterances(u_arr) -> List[str]:
    """兼容 utterances 元素类型：str 或 dict，优先取 'utterance'/'text'。"""
    if isinstance(u_arr, np.ndarray):
        seq = u_arr.tolist()
    else:
        seq = u_arr
    out = []
    for x in seq:
        if isinstance(x, dict):
            s = x.get("utterance") or x.get("text") or ""
        else:
            s = x if isinstance(x, str) else str(x)
        out.append(s)
    return out


def _safe_labels_item(x: Any) -> Dict[str, Any]:
    """labels 可能是 dict/JSON字符串/None -> 统一成 dict。"""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, (np.generic,)):  # numpy 标量
        x = x.item()
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _tokenize_for_bm25(text: str) -> List[str]:
    """中文检索分词（BM25用）。"""
    return list(jieba.cut_for_search(text))


# ========= LLM 关键词提取（面向辩论） =========

# —— 无效/功能性词黑名单（尽量避免出现在关键词里）——
_BAD_WS = {
    "是否", "能否", "可以", "应该", "不应该", "为何", "为什么", "怎样", "怎么", "如何",
    "是否会", "是否应", "是否能", "是否可以", "是不是", "以及", "而且", "因为", "所以",
    "如果", "但是", "但是却", "并且", "或者", "还是", "例如", "比如", "就是说", "那么",
    "问题", "方面", "关系", "影响", "现象", "情况", "观点", "立场", "看法", "理由", "证据",
    "对方", "我方", "辩友", "评委", "观众", "今天", "首先", "其次", "最后", "总之", "综上",
    "我们", "你们", "他们", "大家", "很多", "非常", "特别", "其实", "可能", "一定",
}

# —— 符号/空白清洗 ——
_punct_re = re.compile(r"[\s\u3000\u200b]+")
_non_word_re = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9\-·]+")

def _clean_one_kw(w: str) -> str:
    """基本清洗：去空白/符号，合并连字符，保留中英文数字与常见连接号。"""
    if not w:
        return ""
    w = w.strip()
    w = _punct_re.sub("", w)
    w = _non_word_re.sub("", w)
    # 统一多字符变体
    w = w.replace("--", "-").replace("——", "-").replace("—", "-").replace("·", "·")
    return w

def _is_meaningful_kw(w: str) -> bool:
    """规则：避免虚词、过短、纯数字、全英文功能词；允许 2~12 字的中文名词短语、2~10 长度的英文大写缩写。"""
    if not w:
        return False
    # 黑名单直接剔除
    if w in _BAD_WS:
        return False
    # 过短（单字）且非常见缩写
    if len(w) <= 1:
        # 允许少数有意义单字？一般不建议；如需放开可在此添加白名单
        return False
    # 纯数字/年份常见但一般不直接作为检索关键词
    if w.isdigit():
        return False
    # 过长噪声
    if len(w) > 20:
        return False
    return True

def _llm_chat(messages: List[Dict[str, str]],
              model: str = MODEL_NAME,
              temperature: float = 0.0,
              max_tokens: int = 200) -> Optional[str]:
    """调用OpenAI兼容 /v1/chat/completions，返回 assistant 内容字符串。"""
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    for attempt in range(MAX_RETRY + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            time.sleep(0.6 * (attempt + 1))
        except Exception:
            time.sleep(0.6 * (attempt + 1))
    return None

def _parse_keywords_from_text(s: str) -> List[str]:
    """
    解析 LLM 返回文本中的关键词：
    - 首选 JSON 解析：{"keywords": ["...", "..."]}
    - 否则容错：逗号/换行分割
    - 清洗/过滤无意义词
    """
    if not s:
        return []
    # 尝试JSON
    s_strip = s.strip()
    kws: List[str] = []
    if s_strip.startswith("{") and s_strip.endswith("}"):
        try:
            obj = json.loads(s_strip)
            if isinstance(obj, dict) and "keywords" in obj and isinstance(obj["keywords"], list):
                kws = [str(x) for x in obj["keywords"]]
        except Exception:
            kws = []
    if not kws:
        # 退化：逗号/换行分割
        s_norm = s_strip.replace("，", ",").replace("、", ",")
        parts = []
        for line in s_norm.split("\n"):
            line = line.strip()
            # 去掉“关键词：”“结论：”
            line = line.replace("：", ":")
            if ":" in line:
                line = line.split(":", 1)[-1].strip()
            parts.extend([p.strip() for p in line.split(",") if p.strip()])
        kws = parts

    # 清洗与过滤
    out = []
    seen = set()
    for w in kws:
        w = _clean_one_kw(w)
        if not _is_meaningful_kw(w):
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out

def extract_debate_keywords_via_llm(text: str, k: int) -> List[str]:
    """
    面向“辩论检索”的关键词提取：
    - 关键词类别建议：主题/命题要素（议题、主体、对象）、关键主张与论据支点（因果链环节、评价标准/衡量指标、约束条件）、
      领域术语与实体（政策名、群体/行业名、事件/现象名、理论概念）、动作/策略名（限名词短语）
    - 严禁：功能词、疑问词、纯情绪词、格式话术（是否、为什么、怎么、可以、我们、对方、首先、最后 等）
    - 输出：纯 JSON -> {"keywords": ["...", "..."]}，最多 k 个，按重要性降序
    """
    sys_msg = {
        "role": "system",
        "content": (
            "你是一名“辩论检索关键词工程师”。你的任务是为了在辩论语料库中检索相关论据与证词，"
            "从输入的辩论陈词里提取**可检索的名词短语关键词**（2-4字优先，可包含专有词）。"
            "【必须】避免功能词/疑问词/口水话（如：是否、为什么、怎么、可以、我们、对方、首先、最后、因此、因为、所以 等）。"
            "【必须】聚焦于：议题核心实体（人群/对象/政策/现象）、关键主张（因果要点/评价标准/衡量指标/约束条件）、"
            "领域术语（理论概念/制度名/事件名/专有名词）。"
            "【输出格式】严格返回 JSON：{\"keywords\": [\"...\", \"...\"]}，不包含其他文本。"
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            f"请从以下辩论陈词中抽取**不超过 {k} 个**“可检索的名词短语关键词”，按重要性降序：\n\n{text}\n\n"
            "注意：不要输出功能词/疑问词/口水话；必须是可以在辩论语料里直接检索到的术语或短语。"
        )
    }
    content = _llm_chat([sys_msg, user_msg], model=MODEL_NAME, temperature=0.0, max_tokens=256)
    kws = _parse_keywords_from_text(content)[:k] if content else []
    return kws


# ========= 检索类 =========
@dataclass
class _KwAggCfg:
    per_item_k: int = PER_ITEM_K
    final_kw_k: int = FINAL_TOP_KW


class NpzKeywordListRetriever:
    """
    接受 query_list（list[str]），对每条用 LLM 抽“辩论检索关键词”并聚合，取全局Top-5关键词进行BM25检索（Top-5文段）。
    """
    def __init__(self, npz_path: str, cfg: _KwAggCfg = _KwAggCfg()):
        self.cfg = cfg
        self._load_npz(npz_path)
        self._build_bm25()

    def _load_npz(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"NPZ 文件不存在：{path}")
        data = np.load(path, allow_pickle=True)

        # 必需字段：utterances；可选：labels
        self.utterances = _normalize_utterances(data.get("utterances", []))
        raw_labels = data.get("labels", None)

        # 统一 labels 为 list[dict]，长度与 utterances 对齐
        self.labels: List[Dict[str, Any]] = []
        if raw_labels is None:
            self.labels = [{} for _ in range(len(self.utterances))]
        else:
            items = raw_labels.tolist() if isinstance(raw_labels, np.ndarray) else raw_labels
            for i in range(len(self.utterances)):
                lab = items[i] if isinstance(items, (list, tuple)) and i < len(items) else {}
                self.labels.append(_safe_labels_item(lab))

        if len(self.utterances) == 0:
            raise ValueError("NPZ 中未发现有效的 utterances 用于检索。")

    def _build_bm25(self):
        self.tokenized_corpus = [_tokenize_for_bm25(u) for u in self.utterances]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # ------- 关键词提取与聚合 -------
    def _extract_keywords_single(self, text: str, k: int) -> List[str]:
        """
        优先用“辩论检索”LLM；若失败则退回到 Textrank/词频，并做同样的清洗过滤。
        """
        # 1) LLM（辩论检索提示）
        kws = extract_debate_keywords_via_llm(text, k)
        if kws:
            return kws

        # 2) Textrank
        try:
            raw = jieba.analyse.textrank(text, topK=k) or []
        except Exception:
            raw = []

        # 3) 词频兜底
        if not raw:
            toks = _tokenize_for_bm25(text)
            from collections import Counter
            raw = [w for w, _ in Counter(toks).most_common(k)]

        # 清洗过滤
        out, seen = [], set()
        for w in raw:
            w = _clean_one_kw(w)
            if not _is_meaningful_kw(w):
                continue
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out[:k]

    def _merge_keywords(self, kw_lists: List[List[str]], final_k: int) -> List[str]:
        """
        聚合策略：频次 + 位置权重
        - 频次：某词在多少条子查询中出现
        - 位置：每条里的 rank 采用倒数权重（rank越靠前分越高）
        最后综合分排序，取 Top-K。
        """
        from collections import defaultdict

        freq = defaultdict(int)
        pos_score = defaultdict(float)
        for kws in kw_lists:
            seen_in_this_query = set()
            for rank, w in enumerate(kws):
                w = w.strip()
                if not w:
                    continue
                if w not in seen_in_this_query:
                    freq[w] += 1
                    seen_in_this_query.add(w)
                pos_weight = (self.cfg.per_item_k - rank) / self.cfg.per_item_k
                pos_score[w] += max(pos_weight, 0.0)

        combined = []
        for w in set(list(freq.keys()) + list(pos_score.keys())):
            score = (freq[w] * 10.0) + pos_score[w]
            combined.append((w, score))
        combined.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in combined[:final_k]]

    def extract_global_keywords(self, query_list: List[str]) -> List[str]:
        """对列表每一项提取“辩论检索关键词”并聚合，返回全局Top-K（固定为5）。"""
        if not isinstance(query_list, list) or len(query_list) == 0:
            return []
        per_item_kw = []
        for q in query_list:
            q = q or ""
            per_item_kw.append(self._extract_keywords_single(q, self.cfg.per_item_k))
        top_keywords = self._merge_keywords(per_item_kw, final_k=self.cfg.final_kw_k)
        return top_keywords

    # ------- 检索 -------
    def retrieve_candidates(self, query_list: List[str]) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        输入：query_list（每个元素是一段文本）
        输出：results（list[{"text":..., "labels":...}]), chunk_scores（list[float]）
        """
        final_keywords = self.extract_global_keywords(query_list)
        if not final_keywords:
            # 兜底：用所有子查询拼接做分词作为关键词
            all_text = "\n".join([q or "" for q in query_list])
            final_keywords = _tokenize_for_bm25(all_text)[:FINAL_TOP_KW]

        # 替换 retrieve_candidates 里的这段：
        # 原来：
        # scores = self.bm25.get_scores(final_keywords)

        # 改成：
        # 1) 用与语料相同的分词器把关键词短语切分成 token
        query_tokens = []
        for kw in final_keywords:
            query_tokens.extend(_tokenize_for_bm25(kw))

        # 2) 兜底：如果切完还是空，就把所有子查询拼起来再分词
        if not query_tokens:
            all_text = "\n".join([q or "" for q in query_list])
            query_tokens = _tokenize_for_bm25(all_text)

        # 3) 再去打分
        scores = self.bm25.get_scores(query_tokens)

        idx = np.argsort(scores)[::-1][:BM25_TOPK]

        results: List[Dict[str, Any]] = []
        chunk_scores: List[float] = []
        for i in idx:
            results.append({
                "text": self.utterances[i],
                "labels": self.labels[i] if i < len(self.labels) else {}
            })
            chunk_scores.append(float(scores[i]))
        return results, chunk_scores


# ========= 便捷接口（可在别处直接 import 调用）=========
_retriever_singleton = None
def search(query_list: List[str]) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    便捷函数：固定NPZ/参数，只需传 query_list。
    """
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = NpzKeywordListRetriever(NPZ_PATH)
    return _retriever_singleton.retrieve_candidates(query_list)


# ========= 右键测试入口 =========
def main():
    # 示例：把你的查询列表塞进来即可
    query_list = [
        "我们讨论低欲望社会与主观幸福感之间的关联，应采用哪些衡量指标？",
        "不同年龄段人群在低欲望取向下的生活满意度是否存在显著差异？",
        "若降低外在消费欲望，是否会影响创新动力与长期生产率，从而间接影响幸福？",
        "在政策层面，应关注哪些群体与制度性因素作为评估维度？",
    ]

    # 执行检索
    results, scores = search(query_list)

    # 打印聚合后的Top-5关键词（便于调试/可解释）
    retriever = _retriever_singleton
    top_kws = retriever.extract_global_keywords(query_list)
    print("\n=== 全局Top-5关键词（用于BM25检索） ===")
    print(", ".join(top_kws))

    # 打印结果
    print("\n=== 检索结果（Top-5） ===")
    for i, (item, sc) in enumerate(zip(results, scores), 1):
        print(f"[{i}] score={sc:.4f}")
        print("    text:", item["text"].replace("\n", " "))
        if item["labels"]:
            try:
                pairs = []
                for k, v in item["labels"].items():
                    if isinstance(v, dict) and ("评分" in v or "score" in v):
                        pairs.append(f"{k}({v.get('评分', v.get('score'))})")
                    else:
                        pairs.append(k)
                print("    labels:", ", ".join(pairs))
            except Exception:
                print("    labels: (unprintable)")
    print("\n[OK] 测试完成。")


if __name__ == "__main__":
    main()
