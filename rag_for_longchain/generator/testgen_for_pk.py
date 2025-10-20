# -*- coding: utf-8 -*-
"""
testgen_for_pk.py
- 统一从 config.yaml 读取 openai.api_base
- 只轮换 api_key，不切换 URL
- 5xx/超时自动重试 + 指数退避 + 备用 api_key 切换
- 兼容两种调用方式的 generate_counterargument_via_api（老版本4参 & 新版本命名参数）
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import re
import json
import yaml
import time
import random
import requests
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter, Retry

# ===== 路径（按你现有工程保持不变，可自行修改） =====
CONFIG_FILE = r"../config/config.yaml"
DEBATE_TECHNIQUES_FILE = r"../utils/debate_techniques.json"

# 固定测试文件路径（保留你的写死路径）
TEST_FILE_PATH = (
    r"D:\conversational_rag\rag_for_longchain\data\processed_input\“约辩”第三场澳洲国立大学vs新加坡国立大学_坚持追求真爱是_不是理智的行为\“约辩”第三场澳洲国立大学vs新加坡国立大学_坚持追求真爱是_不是理智的行为.json"
)

# =========================
# 基础加载
# =========================
def load_config(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"配置文件未找到: {file_path}")
        return {}
    except Exception as e:
        print(f"加载配置文件时发生错误: {e}")
        return {}

def load_debate_techniques(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"JSON文件解析错误: {file_path}")
        return {}
    except Exception as e:
        print(f"加载辩论技巧时发生错误: {e}")
        return {}

def load_debate_file(file_path: str) -> Dict[str, Any]:
    """加载用户提供的辩论JSON文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"辩论文件未找到: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"辩论文件解析错误: {file_path}")
        return {}
    except Exception as e:
        print(f"加载辩论文件时发生错误: {e}")
        return {}

CONFIG = load_config(CONFIG_FILE)
DEBATE_TECHNIQUES = load_debate_techniques(DEBATE_TECHNIQUES_FILE)

# =========================
# OpenAI 兼容配置（只用一个 api_base；多 key 轮换）
# =========================
def _read_openai_cfg(cfg: dict) -> Dict[str, Any]:
    """
    支持的 YAML 结构示例：
    openai:
      api_base: "https://api.example.com/v1"   # 主后端（固定不切换）
      llm_model: "gpt-4o"
      # 二选一：
      api_key: "sk-PRIMARY"                    # 单 key
      # 或
      backends:                                # 多 key（推荐）
        - api_key: "sk-PRIMARY"
        - api_key: "sk-BACKUP1"
        - api_key: "sk-BACKUP2"
      # 可选：
      verify_ssl: true
      proxies: null
      headers: {}
      retry_backoff_sec: 1.2
      gateway_rotate_threshold: 5   # 连续多少次 5xx 才换 key
      proxy_rotate_threshold: 3     # 连续多少次连接/代理错误才换 key
      connect_timeout: 10
      read_timeout: 120
    """
    o = (cfg or {}).get("openai", {}) or {}
    api_base = (o.get("api_base") or "").strip()
    model = (o.get("llm_model") or "").strip()
    if not api_base:
        raise RuntimeError("openai.api_base 未配置或为空。")
    if not model:
        raise RuntimeError("openai.llm_model 未配置或为空。")

    # backends: 多密钥（可选）
    backends = o.get("backends")
    if not isinstance(backends, list) or not backends:
        single_key = (o.get("api_key") or "").strip()
        if not single_key:
            raise RuntimeError("未找到 openai.backends，且 openai.api_key 也为空。至少提供一个。")
        backends = [{"api_key": single_key}]

    return {
        "api_base": api_base.rstrip("/"),
        "model": model,
        "backends": backends,
        "verify_ssl": bool(o.get("verify_ssl", True)),
        "headers": o.get("headers") or {},
        "proxies": o.get("proxies"),
        "retry_backoff_sec": float(o.get("retry_backoff_sec", 1.2)),
        "gateway_rotate_threshold": int(o.get("gateway_rotate_threshold", 5)),
        "proxy_rotate_threshold": int(o.get("proxy_rotate_threshold", 3)),
        "connect_timeout": int(o.get("connect_timeout", 10)),
        "read_timeout": int(o.get("read_timeout", 120)),
    }

def _compose_runtime_backend(openai_cfg: Dict[str, Any], backend_idx: int) -> Dict[str, Any]:
    be_list = openai_cfg["backends"]
    b = be_list[backend_idx % len(be_list)]
    api_key = (b.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError(f"openai.backends[{backend_idx}].api_key 为空。")

    # 允许每个 backend 覆盖 proxies/verify_ssl
    proxies = b.get("proxies") if b.get("proxies") is not None else openai_cfg.get("proxies")
    verify_ssl = b.get("verify_ssl") if b.get("verify_ssl") is not None else openai_cfg.get("verify_ssl", True)

    return {
        "api_key": api_key,
        "proxies": proxies,
        "verify_ssl": bool(verify_ssl),
        "headers": openai_cfg.get("headers") or {},
    }

def _build_headers(runtime_cfg: Dict[str, Any]) -> Dict[str, str]:
    """根据当前 key 组装请求头（支持额外自定义 headers 模板）"""
    api_key = runtime_cfg.get("api_key", "")
    base = {"Content-Type": "application/json"}
    extra = runtime_cfg.get("headers") or {}
    merged = dict(base)
    for k, v in extra.items():
        try:
            merged[k] = str(v).format(api_key=api_key)
        except Exception:
            merged[k] = str(v)
    if api_key:
        merged["Authorization"] = f"Bearer {api_key}"
    return merged

def _new_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=0, connect=0, read=0, backoff_factor=0, status_forcelist=[])
    s.mount("http://", HTTPAdapter(max_retries=retries, pool_maxsize=16))
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=16))
    return s

def _shrink_text(txt: str, max_chars: int = 6000) -> str:
    """简单截断（避免过长 prompt 拖垮后端或 413）"""
    if not txt:
        return ""
    if len(txt) > max_chars:
        head = txt[: max_chars // 2]
        tail = txt[-max_chars // 2 :]
        return head + "\n...\n" + tail
    return txt

# =========================
# 请求发送（只轮换 key，不切 URL）
# =========================
def _post_chat_completions(openai_cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    只在同一个 api_base 下工作；不切 URL。
    遇到 5xx（含 504/524）累计到阈值 -> 轮换备用 api_key。
    遇到 401/403/429 -> 立即轮换 api_key。
    遇到代理/连接/读超时累计到阈值 -> 轮换 api_key。
    """
    endpoint = urljoin(openai_cfg["api_base"] + "/", "chat/completions")
    base_sleep = float(openai_cfg.get("retry_backoff_sec", 1.0))

    GATEWAY_ROTATE_THRESHOLD = int(openai_cfg.get("gateway_rotate_threshold", 5))
    PROXY_ROTATE_THRESHOLD   = int(openai_cfg.get("proxy_rotate_threshold", 3))

    backends = openai_cfg["backends"]
    assert isinstance(backends, list) and backends, "openai.backends 至少需要 1 个 api_key"

    backend_idx = 0
    attempt = 0
    consecutive_proxy_fail = 0
    consecutive_gateway_fail = 0

    CONNECT_TO = int(openai_cfg.get("connect_timeout", 10))
    READ_TO    = int(openai_cfg.get("read_timeout", 120))

    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=Retry(total=0)))
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=0)))

    while True:
        attempt += 1
        runtime_cfg = _compose_runtime_backend(openai_cfg, backend_idx)
        headers = _build_headers(runtime_cfg)
        verify_ssl = runtime_cfg.get("verify_ssl", True)
        proxies = runtime_cfg.get("proxies")

        try:
            resp = session.post(
                endpoint,
                headers=headers,
                json=payload,
                verify=verify_ssl,
                proxies=proxies,
                timeout=(CONNECT_TO, READ_TO),
            )

            if 200 <= resp.status_code < 300:
                consecutive_proxy_fail = 0
                consecutive_gateway_fail = 0
                return resp.json()

            # 网关/后端报错（含 504/524）
            if resp.status_code in (502, 503, 504, 524):
                consecutive_gateway_fail += 1
                wait = min(10.0, base_sleep * attempt) + random.uniform(0, 0.5)
                print(f"[警告] backend#{backend_idx} 第 {attempt} 次遇到 HTTP {resp.status_code}，{wait:.1f}s 后重试…")
                if consecutive_gateway_fail >= GATEWAY_ROTATE_THRESHOLD and len(backends) > 1:
                    backend_idx = (backend_idx + 1) % len(backends)
                    consecutive_gateway_fail = 0
                    consecutive_proxy_fail = 0
                    print(f"[提示] 5xx 连续≥{GATEWAY_ROTATE_THRESHOLD}，切换备用 api_key -> backend#{backend_idx}")
                time.sleep(wait)
                continue

            # 鉴权/限流：立即换 key
            if resp.status_code in (401, 403, 429):
                if len(backends) > 1:
                    backend_idx = (backend_idx + 1) % len(backends)
                    consecutive_gateway_fail = 0
                    consecutive_proxy_fail = 0
                    print(f"[提示] 收到 {resp.status_code}，切换备用 api_key -> backend#{backend_idx}")
                    time.sleep(min(5.0, base_sleep * attempt))
                    continue
                else:
                    resp.raise_for_status()

            # 其他 4xx 一般是参数问题
            resp.raise_for_status()

        except (requests.exceptions.ProxyError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            consecutive_proxy_fail += 1
            wait = min(8.0, base_sleep * attempt) + random.uniform(0, 0.5)
            print(f"[警告] backend#{backend_idx} 第 {attempt} 次网络异常：{e.__class__.__name__}，{wait:.1f}s 后重试…")
            if consecutive_proxy_fail >= PROXY_ROTATE_THRESHOLD and len(backends) > 1:
                backend_idx = (backend_idx + 1) % len(backends)
                consecutive_proxy_fail = 0
                consecutive_gateway_fail = 0
                print(f"[提示] 代理/连接连续失败 ≥{PROXY_ROTATE_THRESHOLD} 次，切换备用 api_key -> backend#{backend_idx}")
            time.sleep(wait)
            continue

        except requests.exceptions.SSLError as e:
            if "SSLZeroReturnError" in str(e):
                wait = min(5.0, base_sleep * attempt)
                print(f"[警告] backend#{backend_idx} 第 {attempt} 次遇到 SSLZeroReturnError，{wait:.1f}s 后重试…")
                time.sleep(wait)
                continue
            raise

# =========================
# 文本预处理 & 结构解析
# =========================
def process_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """根据 chunks 汇总文本并选出“最佳辩论技巧”（兼容无 labels）"""
    all_texts = []
    skill_scores: Dict[str, float] = {}

    for chunk in chunks or []:
        all_texts.append(chunk.get("text", "") or chunk.get("utterance", ""))
        labels = chunk.get("labels", {}) or {}
        for skill, data in labels.items():
            score = data.get("评分", data.get("score", 0))
            try:
                score = float(score)
            except Exception:
                score = 0.0
            skill_scores[skill] = skill_scores.get(skill, 0.0) + score

    if not skill_scores:
        best_skill = None
        best_score = 0.0
    else:
        best_skill = max(skill_scores, key=skill_scores.get)
        best_score = skill_scores[best_skill]

    return {"所有文本": all_texts, "最佳辩论技巧": best_skill, "最高评分": best_score}

def parse_debate_data(debate_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容三种来源：
    1) chunks: [{text, labels?}, ...]
    2) debate_history: [{stance, content}, ...]
    3) debate: [{stance, debater, utterance, labels?}, ...]
    """
    topic = debate_data.get("topic", "未指定辩论主题")
    positions = debate_data.get("positions", {"PRO": "正方立场未指定", "CON": "反方立场未指定"})

    chunks = debate_data.get("chunks", []) or []

    pro_history, con_history = [], []

    # 兼容 debate_history
    debate_history = debate_data.get("debate_history", []) or []
    for item in debate_history:
        stance = (item.get("stance") or "").lower()
        content = item.get("content", "") or item.get("utterance", "") or item.get("text", "")
        if stance == "pro":
            pro_history.append(content)
        elif stance == "con":
            con_history.append(content)

    # 兼容 debate
    debate_list = debate_data.get("debate", []) or []
    if debate_list:
        if not chunks:
            chunks = [{"text": it.get("utterance", ""), "labels": it.get("labels", {}) or {}} for it in debate_list]
        for it in debate_list:
            stance = (it.get("stance") or "").lower()
            utt = it.get("utterance", "")
            if stance == "pro":
                pro_history.append(utt)
            elif stance == "con":
                con_history.append(utt)

    return {
        "topic": topic,
        "positions": positions,
        "chunks": chunks,
        "pro_history": "\n".join([t for t in pro_history if t]).strip(),
        "con_history": "\n".join([t for t in con_history if t]).strip(),
    }

def _extract_opposite_debate_text(user_input: dict, stance: str) -> str:
    """从 user_input 中提取与 stance 相反一方的历史文本"""
    s = (stance or "").lower()
    opp = "con" if s == "pro" else "pro"
    txt = user_input.get("con_history") if opp == "con" else user_input.get("pro_history")
    if txt:
        return txt

    # 兜底：从 debate 聚合
    debate = user_input.get("debate") or []
    if debate:
        lines = []
        for it in debate:
            st = (it.get("stance") or "").lower()
            utt = it.get("utterance") or it.get("text") or it.get("content") or ""
            if st == opp and utt:
                lines.append(utt)
        if lines:
            return "\n".join(lines).strip()

    # 再兜底：从 debate_history 聚合
    dh = user_input.get("debate_history") or []
    if dh:
        lines = []
        for it in dh:
            st = (it.get("stance") or "").lower()
            cont = it.get("content") or it.get("utterance") or it.get("text") or ""
            if st == opp and cont:
                lines.append(cont)
        if lines:
            return "\n".join(lines).strip()

    return ""

# =========================
# 逻辑漏洞分析（可选）
# =========================
def analyze_logical_fallacies(debate_text: str, topic: str, opposite_stance: str) -> str:
    try:
        openai_cfg = _read_openai_cfg(CONFIG)
        payload = {
            "model": openai_cfg["model"],
            "stream": False,
            "messages": [
                {"role": "system", "content": "你是一位逻辑分析专家，擅长识别辩论中的逻辑漏洞和谬误。"},
                {"role": "user", "content":
                    f"""请分析以下辩论文本中可能存在的逻辑漏洞。
辩论主题：{topic}
对方立场：{opposite_stance}
辩论文本：{debate_text}

请以自然语言详细指出文本中的一到两条逻辑漏洞，注意：只能生成一条或者两条逻辑错误，包括但不限于：
1. 前提假设的不合理之处
2. 推理过程中的逻辑谬误
3. 结论与论据之间的矛盾
4. 论据本身的真实性或相关性问题

请用清晰、有条理的方式列出这些逻辑漏洞，不需要使用任何JSON格式，直接以段落或分点形式呈现。"""
                }
            ]
        }
        result_json = _post_chat_completions(openai_cfg, payload)
        if result_json and "choices" in result_json:
            return result_json["choices"][0]["message"]["content"]
        return "（未检出明显逻辑漏洞）"
    except Exception as e:
        print(f"[逻辑分析] 调用出错：{e}")
        return "（逻辑漏洞分析时发生错误）"

# =========================
# Prompt 组装
# =========================
def generate_prompt(all_chunks: str,
                    all_texts_variable: str,
                    technique: Optional[str],
                    technique_details: Dict[str, Any],
                    stance: str,
                    position_text: str,
                    opposite_debate_text: str,
                    topic: str) -> str:
    stance_cn = "正方（pro）" if (stance or "").lower() == "pro" else "反方（con）"
    opposite_stance_cn = "反方（con）" if (stance or "").lower() == "pro" else "正方（pro）"

    logical_fallacies_text = analyze_logical_fallacies(
        debate_text=opposite_debate_text,
        topic=topic,
        opposite_stance=opposite_stance_cn
    )
    logical_fallacies_info = f"对方辩手（{opposite_stance_cn}）的辩论文本中存在以下逻辑漏洞：\n{logical_fallacies_text}\n"

    prompt = f"""
以下是用户输入的内容：
{all_chunks}

以及检索到的文本：
{all_texts_variable}

使用的辩论技巧是：{technique or '（未指定）'}
辩论技巧的定义是：{technique_details.get("定义", "")}
相关背景是：{technique_details.get("场景背景", "")}
示例文本是：{technique_details.get("示例文本", "")}

你的立场是：{stance_cn}
你代表的具体立场是：“{position_text}”
你是一位从业20年的辩论专家。请你扮演处于赛场的一位辩手，你的目标是取得辩论赛的胜利，你需要坚定地从“{stance_cn}”出发，生成一段处于“{stance_cn}”的辩论陈词，对持相反立场选手的发言进行有力回怼。

特别注意对方辩手的逻辑漏洞，并在回怼中针对这些漏洞进行反驳：
{logical_fallacies_info}
""".strip()
    return prompt

# =========================
# 统一：生成回怼（兼容老/新两种调用方式）
# =========================
def _normalize_inputs_for_generation(*args, **kwargs) -> Tuple[Dict[str, Any], str, Dict[str, Any], str]:
    """
    返回统一四元组：
    (user_input, all_texts_variable, process_result, stance)
    支持两种调用：
    1) 老版（4个位置参数）：
       generate_counterargument_via_api(debate_json, retrieved_text, {"最佳辩论技巧": best_skill}, stance)
    2) 新版（命名参数）：
       generate_counterargument_via_api(user_input=..., all_texts_variable=..., result=..., stance="pro")
    """
    if kwargs:
        user_input = kwargs.get("user_input") or kwargs.get("debate_json") or (args[0] if args else {})
        all_texts_variable = kwargs.get("all_texts_variable") or kwargs.get("retrieved_text") or (args[1] if len(args) > 1 else "")
        process_result = kwargs.get("result") or kwargs.get("info") or (args[2] if len(args) > 2 else {})
        stance = kwargs.get("stance") or (args[3] if len(args) > 3 else "pro")
        return user_input, str(all_texts_variable), process_result or {}, str(stance)

    # 无 kwargs：按老版位置参数解析
    if len(args) >= 4 and isinstance(args[0], dict):
        user_input = args[0]
        all_texts_variable = str(args[1])
        process_result = args[2] if isinstance(args[2], dict) else {}
        stance = str(args[3])
        return user_input, all_texts_variable, process_result, stance

    raise TypeError("generate_counterargument_via_api 参数不匹配。应为4个位置参数或显式命名参数。")

def generate_counterargument_via_api(*args, **kwargs) -> str:
    """
    生成回怼发言（字符串）。
    - 自动读取 config.yaml 中的 openai.api_base / backends。
    - 对 5xx / 超时 / 连接错误 自动重试与切换 api_key（URL 不变）。
    """
    user_input, all_texts_variable, process_result, stance = _normalize_inputs_for_generation(*args, **kwargs)
    parsed = parse_debate_data(user_input)

    topic = parsed["topic"]
    positions = parsed["positions"] or {"PRO": "", "CON": ""}
    position_text = positions.get("PRO" if (stance or "").lower() == "pro" else "CON", "")
    technique = process_result.get("最佳辩论技巧")
    technique_details = DEBATE_TECHNIQUES.get(technique, {}) if technique else {}
    opposite_text = _extract_opposite_debate_text(
        {
            **user_input,
            "pro_history": parsed.get("pro_history", ""),
            "con_history": parsed.get("con_history", ""),
        },
        stance=stance,
    )

    prompt = generate_prompt(
        all_chunks="\n".join([t for t in process_result.get("所有文本", []) if t]),
        all_texts_variable=all_texts_variable,
        technique=technique,
        technique_details=technique_details,
        stance=stance,
        position_text=position_text,
        opposite_debate_text=opposite_text,
        topic=topic,
    )

    openai_cfg = _read_openai_cfg(CONFIG)
    payload = {
        "model": openai_cfg["model"],
        "stream": False,
        "messages": [
            {"role": "system", "content": "你是一位资深辩手，语言凝练、逻辑严密、攻击性强，避免客套。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    data = _post_chat_completions(openai_cfg, payload)
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return text.strip()

# =========================
# 主函数（便于右键直接运行做本地测试）
# =========================
def main():
    file_path = TEST_FILE_PATH
    stance = "pro"  # 测试时可改为 "con"

    print("===== 开始辩论系统 =====")
    print(f"加载辩论文件: {file_path}")
    print(f"我方立场: {'正方' if stance == 'pro' else '反方'}\n")

    debate_data = load_debate_file(file_path)
    if not debate_data:
        print("无法加载辩论数据，程序退出")
        return

    parsed_data = parse_debate_data(debate_data)
    print(f"辩论主题: {parsed_data['topic']}")
    print(f"正方立场: {parsed_data['positions'].get('PRO')}")
    print(f"反方立场: {parsed_data['positions'].get('CON')}\n")

    print("===== 处理文本块并分析最佳辩论技巧 =====")
    process_result = process_chunks(parsed_data["chunks"])
    if process_result['最佳辩论技巧'] is None:
        print("未在数据中发现任何技巧评分，将使用默认技巧进行生成。")
    print(f"最佳辩论技巧: {process_result['最佳辩论技巧']}")
    print(f"最高评分: {process_result['最高评分']}\n")

    user_input_for_gen = {
        "positions": parsed_data["positions"],
        "topic": parsed_data["topic"],
        "pro_history": parsed_data["pro_history"],
        "con_history": parsed_data["con_history"],
        "debate": debate_data.get("debate", []),
        "debate_history": debate_data.get("debate_history", []),
    }

    print("===== 对方辩论历史（自动抽取） =====")
    preview_opp = _extract_opposite_debate_text(user_input_for_gen, stance)
    print(preview_opp[:500] + "..." if len(preview_opp) > 500 else (preview_opp or "(空)"))
    print()

    print("===== 生成回怼内容 =====")
    counterargument = generate_counterargument_via_api(
        user_input=user_input_for_gen,
        all_texts_variable="\n".join(process_result["所有文本"]),
        result=process_result,
        stance=stance,
    )

    print("\n===== 最终回怼内容 =====")
    print(counterargument)
    print("\n===== 处理结束 =====")


if __name__ == "__main__":
    main()
