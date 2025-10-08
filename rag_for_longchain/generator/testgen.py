# -*- coding: utf-8 -*-
from typing import List, Dict, Any
import requests
import yaml
import json
import time
import random
from urllib.parse import urljoin

# === 路径 ===
CONFIG_FILE = "D:/converstional_rag/rag_for_longchain/config/config.yaml"
DEBATE_TECHNIQUES_FILE = "D:/converstional_rag/rag_for_longchain/utils/debate_techniques.json"


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


config = load_config(CONFIG_FILE)
DEBATE_TECHNIQUES = load_debate_techniques(DEBATE_TECHNIQUES_FILE)


# =========================
# OpenAI 兼容配置读取
# =========================
def _read_openai_cfg(cfg: dict) -> Dict[str, Any]:
    """
    期望 YAML 结构：
    openai:
      api_key: "sk-xxxx"
      api_base: "https://www.xdaicn.top/v1"
      llm_model: "gpt-5"
      # 可选：
      # verify_ssl: true
      # proxies: {"http": "...", "https": "..."}
      # headers: {"X-Custom": "foo"}
      # retry_backoff_sec: 1.0
      # max_tokens: 512
      # temperature: 0.7
    """
    o = (cfg or {}).get("openai", {}) or {}
    api_key = (o.get("api_key") or "").strip()
    api_base = (o.get("api_base") or "").strip()
    llm_model = (o.get("llm_model") or "").strip()
    if not api_key:
        raise RuntimeError("openai.api_key 未配置或为空。")
    if not api_base:
        raise RuntimeError("openai.api_base 未配置或为空。")
    if not llm_model:
        raise RuntimeError("openai.llm_model 未配置或为空。")

    return {
        "api_key": api_key,
        "api_base": api_base.rstrip("/"),
        "model": llm_model,
        "verify_ssl": bool(o.get("verify_ssl", True)),
        "proxies": o.get("proxies"),
        "headers": o.get("headers") or {},
        "retry_backoff_sec": float(o.get("retry_backoff_sec", 1.0)),
        "max_tokens": o.get("max_tokens"),   # 可为 None
        "temperature": o.get("temperature"), # 可为 None
    }


def _build_headers(openai_cfg: Dict[str, Any]) -> Dict[str, str]:
    api_key = openai_cfg.get("api_key", "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "",
    }
    extra = openai_cfg.get("headers") or {}
    merged = {**headers}
    for k, v in extra.items():
        try:
            merged[k] = str(v).format(api_key=api_key)
        except Exception:
            merged[k] = str(v)
    if not api_key and "Authorization" in merged:
        merged.pop("Authorization", None)
    return merged


# =========================
# 请求发送（带无限重试）
# =========================
def _post_chat_completions(openai_cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    无限重试的条件：
      - SSLError 且包含 SSLZeroReturnError
      - HTTP 524 / 502 / 503 / 504
    其余错误直接抛出。
    不设置 timeout。
    """
    endpoint = urljoin(openai_cfg["api_base"] + "/", "chat/completions")
    headers = _build_headers(openai_cfg)
    verify_ssl = openai_cfg.get("verify_ssl", True)
    proxies = openai_cfg.get("proxies")
    base_sleep = float(openai_cfg.get("retry_backoff_sec", 1.0))

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                verify=verify_ssl,
                proxies=proxies,
            )
            if 200 <= resp.status_code < 300:
                return resp.json()

            if resp.status_code in (524, 502, 503, 504):
                sleep = min(10.0, base_sleep * attempt) + random.uniform(0, 0.5)
                print(f"[警告] 第 {attempt} 次遇到 HTTP {resp.status_code}（上游超时/暂不可用），{sleep:.1f}s 后重试…")
                time.sleep(sleep)
                continue

            # 其它非 2xx：抛出以便上层感知
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.SSLError as e:
            if "SSLZeroReturnError" in str(e):
                sleep = min(5.0, base_sleep * attempt)
                print(f"[警告] 第 {attempt} 次遇到 SSLZeroReturnError，{sleep:.1f}s 后重试…")
                time.sleep(sleep)
                continue
            # 其它 SSL 错误直接抛出
            raise

        except requests.exceptions.RequestException:
            # 其它网络/HTTP错误直接抛出
            raise


# =========================
# 业务：生成提示词
# =========================
def generate_prompt(all_chunks, all_texts_variable, technique, technique_details, stance, position_text):
    stance_cn = "正方（pro）" if stance == "pro" else "反方（con）"
    prompt = f'''
以下是用户输入的内容：
{all_chunks}

以及检索到的文本：
{all_texts_variable}

使用的辩论技巧是：{technique}
辩论技巧的定义是：{technique_details.get("定义", "")}
相关背景是：{technique_details.get("场景背景", "")}
示例文本是：{technique_details.get("示例文本", "")}

你的立场是：{stance_cn}
你代表的具体立场是：“{position_text}”
你是一位从业20年的辩论专家。请你扮演处于赛场的一位辩手，你的目标是取得辩论赛的胜利，你需要坚定地从“{stance_cn}”出发，生成一段处于“{stance_cn}”的辩论陈词，对持相反立场选手的发言进行有力回怼。
    '''
    return prompt.strip()


# =========================
# 业务：调用 LLM 生成回怼
# =========================
def generate_counterargument_via_api(
    user_input: dict,
    all_texts_variable: str,
    result: dict,
    stance: str,  # "pro" or "con"
) -> str:
    pos = user_input.get("positions") or {}
    key = stance.upper()
    position_text = pos.get(key, "")

    technique = result.get("最佳辩论技巧")
    if not technique or technique not in DEBATE_TECHNIQUES:
        return f"未找到指定的辩论技巧：{technique}，请提供有效的技巧。"
    technique_details = DEBATE_TECHNIQUES[technique]

    # 读取 openai 节点配置（与你的 YAML 完全一致）
    try:
        openai_cfg = _read_openai_cfg(config)
    except Exception as e:
        return f"OpenAI 配置错误：{e}"

    model_name = openai_cfg.get("model")
    if not model_name:
        return "openai.llm_model 未配置。"

    prompt = generate_prompt(
        user_input,
        all_texts_variable,
        technique,
        technique_details,
        stance,
        position_text
    )
    print("我的提示：", prompt)

    # 构造 payload；允许从 YAML 读取 max_tokens/temperature
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    if openai_cfg.get("max_tokens") is not None:
        payload["max_tokens"] = int(openai_cfg["max_tokens"])
    if openai_cfg.get("temperature") is not None:
        payload["temperature"] = float(openai_cfg["temperature"])

    try:
        result_json = _post_chat_completions(openai_cfg, payload)
        return result_json["choices"][0]["message"]["content"]
    except requests.exceptions.SSLError as e:
        return f"调用API时发生 SSL 错误：{e}"
    except requests.exceptions.RequestException as e:
        return f"调用API时发生网络错误：{e}"
    except Exception as e:
        return f"调用API时发生未知错误：{e}"


# =========================
# 业务：处理 chunks，选最佳技巧
# =========================
def process_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_texts = []
    skill_scores: Dict[str, float] = {}

    for chunk in chunks:
        all_texts.append(chunk.get("text", ""))
        labels = chunk.get("labels", {}) or {}
        for skill, data in labels.items():
            score = data.get("评分", 0)
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

    return {
        "所有文本": all_texts,
        "最佳辩论技巧": best_skill,
        "最高评分": best_score
    }
