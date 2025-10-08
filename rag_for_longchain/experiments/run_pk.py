# -*- coding: utf-8 -*-
# @Time    : 2025/7/15 21:31
# @Author  : Maoyuan Li
# @File    : run_pk.py
# @Software: PyCharm

import os
import json
import yaml
import re
import time
import requests
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy
from transformers import AutoTokenizer, AutoModel
from typing import Optional

def append_jsonl(path: Path, obj: dict) -> None:
    """向 JSONL 追加一行，立刻落盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write('\n')
        f.flush()  # 立即写盘

def write_progress(progress_file: Path, idx: int, total: int, subdir: Path, extra: str = "") -> None:
    """把当前进度写到文本文件，便于你快速定位。"""
    content = f"processed={idx}/{total}\ncurrent_dir={str(subdir)}\n{extra}".strip()
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(content, encoding="utf-8")

# 如果你的文件里用到了这个函数签名，请把'| None'改成 Optional[str]
def _safe_load_model(model_name: str, local_dir: Optional[str] = None):
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        return tok, mdl
    except Exception as e:
        if local_dir and os.path.isdir(local_dir):
            tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
            mdl = AutoModel.from_pretrained(local_dir,  local_files_only=True)
            return tok, mdl
        raise RuntimeError(
            f"[HF加载失败] {model_name} 下载失败且无可用本地目录: {local_dir}\n"
            f"原始异常: {repr(e)}"
        ) from e



# ========= 运行稳定性（权宜） =========
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"       # 先跑起来，避免 OpenMP 冲突
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"  # 减少线程层冲突

# ========= HuggingFace/网络稳态设置 =========
# 更大的超时 + Rust 传输（更稳）
os.environ.setdefault("HF_HUB_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# 为 huggingface 域名走直连（如果你有系统代理且它不稳定，能缓解 TLS 中断）
os.environ.setdefault("NO_PROXY", "huggingface.co,cdn-lfs.huggingface.co")

# 将证书固定为 certifi（绕开系统证书混乱）
try:
    import certifi
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

# （可选）让 transformers 优先读本地缓存（如果你已经把模型下载到本地）
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ------------------ 配置加载 ------------------
CONFIG_FILE = Path("D:/converstional_rag/rag_for_longchain/config/config.yaml")

def load_config(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

cfg       = load_config(CONFIG_FILE)
api_key   = cfg.get('openai', {}).get('api_key', '')
api_base  = cfg.get('openai', {}).get('api_base', '')
llm_model = cfg.get('openai', {}).get('llm_model', 'gpt-4o')

# ------------------ 通用：安全加载 HF 模型（在线->本地兜底） ------------------
from typing import Optional

def _safe_load_model(model_name: str, local_dir: Optional[str] = None):

    """
    先尝试在线加载；失败则回退到 local_dir（若存在）。
    返回: (tokenizer, model)
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        return tok, mdl
    except Exception as e:
        if local_dir and os.path.isdir(local_dir):
            tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
            mdl = AutoModel.from_pretrained(local_dir,  local_files_only=True)
            return tok, mdl
        # 抛出更友好的错误说明
        raise RuntimeError(
            f"[HF加载失败] 无法下载 {model_name} 且未找到可用的本地目录: {local_dir}\n"
            f"原始异常: {repr(e)}\n"
            f"=> 解决方案：用 huggingface-cli 先把模型拉到本地，然后把 ENC_LOCAL_DIR 指向那个目录。"
        ) from e

# ------------------ Utterance 提取 ------------------
def extract_utterances(data):
    utterances = []
    if isinstance(data, str):
        utterances += [line.strip() for line in data.splitlines() if line.strip()]
    elif isinstance(data, dict):
        for k, v in data.items():
            if k.lower() in ('utterance', 'utterances'):
                if isinstance(v, list):
                    utterances += [str(x) for x in v]
                else:
                    utterances.append(str(v))
            else:
                utterances += extract_utterances(v)
    elif isinstance(data, list):
        for item in data:
            utterances += extract_utterances(item)
    return utterances

def prepare_query(path_or_obj):
    if isinstance(path_or_obj, (str, Path)) and str(path_or_obj).lower().endswith('.json') and Path(path_or_obj).is_file():
        obj = json.loads(Path(path_or_obj).read_text(encoding="utf-8"))
    else:
        obj = path_or_obj
    return "\n".join(extract_utterances(obj))

# ------------------ Naive RAG 内部函数 ------------------
def _post_with_retry(url, headers, payload, timeout=60, max_tries=2):
    last_err = None
    for i in range(max_tries):
        try:
            return requests.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception as e:
            last_err = e
            # 简单指数退避
            time.sleep(1.5 * (i + 1))
    raise last_err

def generate_side_via_api(title, texts, retrieved_utts, want_stance: str):
    url = api_base.rstrip('/') + '/chat/completions'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    meta = json.dumps({'title': title, 'stance': want_stance}, ensure_ascii=False)
    context = '\n'.join(f"Chunk {i+1}: {u}" for i, u in enumerate(retrieved_utts))
    prompt = f"""请仅生成{ '正方(pro)' if want_stance.lower()=='pro' else '反方(con)' }的一段辩论发言，<=1100字。
返回**严格 JSON**：{{"title": "...", "model": "...", "speech": "..."}}。

辩论元数据：
{meta}

用户输入：
{texts}

检索材料：
{context}
"""

    payload = {
        'model': llm_model,
        'messages': [
            {'role': 'system', 'content': 'You are a debate assistant. Only output JSON.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7,
        'max_tokens': 2048,
        # 如果你的代理支持 OpenAI 的 JSON 模式，强烈建议开启 ↓↓↓
        'response_format': {'type': 'json_object'}
    }

    resp = _post_with_retry(url, headers, payload, timeout=60, max_tries=2)
    resp.raise_for_status()
    data = resp.json()
    raw = data['choices'][0]['message']['content']

    # 保险：先尝试直接 loads；失败就提取首个 JSON 子串再 loads
    try:
        out = json.loads(raw)
    except Exception:
        m = re.search(r'\{[\s\S]*\}', raw)  # 贪婪匹配第一段花括号 JSON
        out = json.loads(m.group(0)) if m else {"title": title, "model": llm_model, "speech": raw.strip()}

    # 补齐字段 + 去空白
    out.setdefault("title", title)
    out.setdefault("model", llm_model)
    out["speech"] = (out.get("speech") or "").strip()
    return out

# ------------------ 检索器 ------------------
class Retriever:
    def __init__(self, db_path, encoder_name, max_tokens=512, local_dir=None):
        data  = np.load(db_path, allow_pickle=True)
        files = set(data.files)
        # 加载 utterances（字符串数组）
        if 'utterances' in files:
            self.utterances = data['utterances']
        elif 'utterance' in files:
            self.utterances = data['utterance']
        else:
            self.utterances = next(
                data[n] for n in data.files
                if data[n].dtype == object or data[n].dtype.kind in ('U', 'S')
            )
        # 加载 embeddings（数值数组）
        if 'embeddings' in files:
            self.embeddings = data['embeddings']
        elif 'embedding' in files:
            self.embeddings = data['embedding']
        else:
            self.embeddings = next(
                data[n] for n in data.files
                if np.issubdtype(data[n].dtype, np.number)
            )
        # 初始化编码器（在线->本地兜底）
        self.tokenizer, self.model = _safe_load_model(encoder_name, local_dir=local_dir)
        self.max_tokens = max_tokens
        # 归一化
        self.embeddings = self.embeddings.astype(float)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        self.embeddings /= norms

    def _embed(self, text):
        toks = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_tokens
        )
        with torch.no_grad():
            out = self.model(**toks)
        vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-10)

    def retrieve(self, json_path, top_k=5):
        txt = prepare_query(json_path)
        emb = self._embed(txt)
        sims = self.embeddings.dot(emb)
        idx  = np.argsort(sims)[-top_k:][::-1]
        return [self.utterances[i] for i in idx]

# ------------------ 固定立场包装器 ------------------
class FixedStanceModel:
    def __init__(self, model, fixed_stance):
        self._inner     = model
        self._stance    = fixed_stance.lower()
        self._orig_name = getattr(model, 'name', type(model).__name__)
    @property
    def name(self):
        return f"{self._orig_name}({self._stance})"
    def generate_utterance(self, debate_json, context, stance):
        # 直接用固定立场，无视传入的 stance
        return self._inner.generate_utterance(debate_json, context, self._stance)

# ------------------ NaiveRAGModel 适配 ------------------
class NaiveRAGModel:
    def __init__(self, json_path, retriever, top_k=5):
        self.json_path = json_path
        self.title     = Path(json_path).stem
        self.retriever = retriever
        self.top_k     = top_k
    @property
    def name(self):
        return "NaiveRAG"

    def generate_utterance(self, debate_json, context, stance):
        retrieved = self.retriever.retrieve(self.json_path, top_k=self.top_k)
        texts     = prepare_query(debate_json)
        out       = generate_side_via_api(self.title, texts, retrieved, want_stance=stance)
        speech    = out.get("speech", "").strip()

        # 兜底：如果仍然空，尝试一次“再生成”提示
        if not speech:
            retry_prompt_suffix = "请用更凝练的方式给出一段完整、可上场的辩论发言。仍然只返回 JSON。"
            out2 = generate_side_via_api(self.title, texts + "\n" + retry_prompt_suffix, retrieved, want_stance=stance)
            speech = (out2.get("speech") or "").strip()

        # 再兜底：仍为空，就返回一个可见的占位信息，便于定位问题
        return speech or "[NaiveRAG未生成有效内容]"

# ===== 外部依赖（RDebaterModel / PKManager） =====
# 注意：RDebaterModel 内部如果也从 HF 下载模型，
# 也会受上面环境变量与证书设置的正向影响。
from rag_for_longchain.experiments.Rdebater_for_pk import RDebaterModel
from pkmanager import PKManager

# ------------------ 主流程 ------------------
# ==== 主流程（__main__）替换为下方版本 ====
if __name__ == '__main__':
    INPUT_DIR    = Path(r"D:\converstional_rag\23acldata\input_data\processed_input")
    OUTPUT_DIR   = Path(r"D:\converstional_rag\23acldata\output_data\pk_result")
    OUTPUT_JSONL = OUTPUT_DIR / "debate_pk_results_gpt-4o_1.jsonl"   # 每场一行
    OUTPUT_SNAPSHOT = OUTPUT_DIR / "debate_pk_results_snapshot_1.json" # 每若干场覆盖写一份快照（可选）
    PROGRESS_FILE   = OUTPUT_DIR / "progress.txt"

    DB_PATH      = r"D:/converstional_rag/rag_for_longchain/experiments/structured_embeddings_for_naive_rag.npz"
    ENC_MODEL    = "hfl/chinese-roberta-wwm-ext-large"
    ENC_LOCAL_DIR = r"D:\hf_models\hfl-chinese-roberta-wwm-ext-large"  # 你的本地模型目录

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 枚举比赛目录
    subdirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    total = len(subdirs)

    # 单例检索器（带本地兜底）
    retriever = Retriever(DB_PATH, ENC_MODEL, local_dir=ENC_LOCAL_DIR)
    all_results = []  # 仅用于周期性快照

    # 逐场处理
    for idx, sub in enumerate(subdirs, start=1):
        try:
            # 找到该场比赛 JSON（排除 last_two.json）
            json_files = [p for p in sub.glob("*.json") if p.name != 'last_two.json']
            if not json_files:
                msg = f"[跳过] 目录无可用 JSON: {sub}"
                print(msg)
                write_progress(PROGRESS_FILE, idx-1, total, sub, extra=msg)
                continue
            fn = json_files[0]

            # 读取原始 JSON
            try:
                orig_json = json.loads(fn.read_text(encoding="utf-8"))
            except Exception as e:
                msg = f"[跳过] 读取 JSON 失败: {fn} | {e}"
                print(msg)
                write_progress(PROGRESS_FILE, idx-1, total, sub, extra=msg)
                continue

            # 深拷贝，保证每场 debate 独立
            debate_copy = deepcopy(orig_json)

            # 实例化各模型
            rdebater = RDebaterModel(str(CONFIG_FILE))
            naive    = NaiveRAGModel(str(fn), retriever)

            # 固定立场：RDebater 只做 pro，NaiveRAG 只做 con
            pro_model = FixedStanceModel(rdebater, 'pro')
            con_model = FixedStanceModel(naive,    'con')

            # 多轮对抗
            manager = PKManager(pro_model=pro_model, con_model=con_model, max_rounds=3)
            full_debate = manager.run_debate(debate_copy)

            # === 关键：增量写入 ===
            record = {
                "index": idx,
                "dir": str(sub),
                "file": str(fn),
                "result": full_debate
            }
            append_jsonl(OUTPUT_JSONL, record)   # 立刻落盘一场
            all_results.append(record)

            # 每处理若干场就写一个快照（可选，防止只剩 JSONL）
            if idx % 5 == 0 or idx == total:
                OUTPUT_SNAPSHOT.write_text(
                    json.dumps(all_results, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

            # 更新进度文件
            write_progress(PROGRESS_FILE, idx, total, sub, extra="ok")

            print(f"✅ 已完成第 {idx}/{total} 场：{sub.name}")

        except Exception as e:
            # 捕获“该场比赛”的任何关键异常，写进度并中止
            err = f"[中止] 第 {idx}/{total} 场失败：{sub} | {repr(e)}"
            print(err)
            write_progress(PROGRESS_FILE, idx-1, total, sub, extra=err)
            break

    print(f"👉 进度文件: {PROGRESS_FILE}")
    print(f"👉 增量结果(JSONL): {OUTPUT_JSONL}")
    print(f"👉 快照(JSON数组): {OUTPUT_SNAPSHOT}")
