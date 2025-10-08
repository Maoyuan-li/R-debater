# -*- coding: utf-8 -*-
# @Time    : 2025/10/2 17:51
# @Author  : Maoyuan Li
# @File    : keyword_retriever.py.py
# @Software: PyCharm
"""
薄封装：复用你现成的 npz_keyword_retriever（关键词检索BM25）
要求：与 faiss_retriever.py 同目录
"""

from typing import List, Tuple, Dict, Any
try:
    # 你的那份文件名（保持与上传一致，或改名成这个）
    from .npz_keyword_retriever import NpzKeywordListRetriever, _KwAggCfg
except Exception:
    # 兼容：如果你是直接把 npz_keyword_retriever.py 放同目录，不带包导入
    from npz_keyword_retriever import NpzKeywordListRetriever, _KwAggCfg  # type: ignore

# 你自己的 all.npz 路径；与 npz_keyword_retriever.py 内保持一致也可
NPZ_PATH = r"D:\converstional_rag\rag_for_longchain\data\processed_data\allnpz\all.npz"

# 单例
_retriever = None

def _get_retriever() -> NpzKeywordListRetriever:
    global _retrlver, _retriever
    if _retriever is None:
        _retriever = NpzKeywordListRetriever(npz_path=NPZ_PATH, cfg=_KwAggCfg())
    return _retriever

def search(query_list: List[str]) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    统一检索接口：输入 query_list -> 返回 (results, scores)
    results: [{"text": ..., "labels": {...}}, ...], scores: [float, ...]
    """
    return _get_retriever().retrieve_candidates(query_list)