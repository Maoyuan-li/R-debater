# -*- coding: utf-8 -*-
from typing import List, Dict, Any
import requests
import yaml
import json
import time
import random
from urllib.parse import urljoin

# === Paths ===
CONFIG_FILE = "D:\\conversational_rag/rag_for_longchain\\config\\config.yaml"
DEBATE_TECHNIQUES_FILE = "D:\\conversational_rag/rag_for_longchain/utils\\debate_techniques.json"


# =========================
# Basic loading
# =========================
def load_config(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Config file not found: {file_path}")
        return {}
    except Exception as e:
        print(f"Error occurred while loading config: {e}")
        return {}


def load_debate_techniques(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"JSON parse error: {file_path}")
        return {}
    except Exception as e:
        print(f"Error occurred while loading debate techniques: {e}")
        return {}


config = load_config(CONFIG_FILE)
DEBATE_TECHNIQUES = load_debate_techniques(DEBATE_TECHNIQUES_FILE)


# =========================
# OpenAI-compatible config reader
# =========================
def _read_openai_cfg(cfg: dict) -> Dict[str, Any]:
    """
    Expected YAML structure:
    openai:
      api_key: "sk-xxxx"
      api_base: "https://www.xdaicn.top/v1"
      llm_model: "gpt-5"
      # Optional:
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
        raise RuntimeError("openai.api_key is not configured or empty.")
    if not api_base:
        raise RuntimeError("openai.api_base is not configured or empty.")
    if not llm_model:
        raise RuntimeError("openai.llm_model is not configured or empty.")

    return {
        "api_key": api_key,
        "api_base": api_base.rstrip("/"),
        "model": llm_model,
        "verify_ssl": bool(o.get("verify_ssl", True)),
        "proxies": o.get("proxies"),
        "headers": o.get("headers") or {},
        "retry_backoff_sec": float(o.get("retry_backoff_sec", 1.0)),
        "max_tokens": o.get("max_tokens"),   # can be None
        "temperature": o.get("temperature"), # can be None
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
# Request sender (with infinite retry for specific cases)
# =========================
def _post_chat_completions(openai_cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infinite retry conditions:
      - SSLError containing SSLZeroReturnError
      - HTTP 524 / 502 / 503 / 504
    Other errors are raised directly.
    No timeout is set.
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
                print(f"[Warning] Attempt {attempt}: HTTP {resp.status_code} (upstream timeout / temporarily unavailable). Retrying in {sleep:.1f}s…")
                time.sleep(sleep)
                continue

            # Other non-2xx: raise for upper layer to handle
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.SSLError as e:
            if "SSLZeroReturnError" in str(e):
                sleep = min(5.0, base_sleep * attempt)
                print(f"[Warning] Attempt {attempt}: SSLZeroReturnError. Retrying in {sleep:.1f}s…")
                time.sleep(sleep)
                continue
            # Other SSL errors: raise
            raise

        except requests.exceptions.RequestException:
            # Other network/HTTP errors: raise
            raise


# =========================
# Business: prompt construction
# =========================
def _get_with_fallback(d: Dict[str, Any], *keys, default: str = "") -> str:
    """Try a sequence of keys (english first, then chinese) and return first non-empty string."""
    for k in keys:
        v = d.get(k, "")
        if isinstance(v, str) and v.strip():
            return v
    return default

def generate_prompt(all_chunks, all_texts_variable, technique, technique_details, stance, position_text):
    stance_str = "Pro" if stance == "pro" else "Con"

    definition = _get_with_fallback(technique_details, "definition", "定义")
    background = _get_with_fallback(technique_details, "background", "场景背景")
    example = _get_with_fallback(technique_details, "example", "示例文本")

    prompt = f"""
Below is the user's input:
{all_chunks}

And the retrieved texts:
{all_texts_variable}

Selected debate technique: {technique}
Technique definition: {definition}
Context/background: {background}
Example text: {example}

Your stance: {stance_str}
Your specific position is: "{position_text}"
You are a debate expert with 20 years of experience. Role-play as a contestant in a formal debate. 
Your goal is to win: generate a strong speech from the "{stance_str}" stance and **refute** the opponent's claims effectively.
Output **everything** in English.
""".strip()
    return prompt


# =========================
# Business: call LLM to generate counterargument
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

    # Support both english and chinese keys for compatibility
    technique = result.get("best_technique") or result.get("最佳辩论技巧")
    if not technique or technique not in DEBATE_TECHNIQUES:
        return f"Specified debate technique not found: {technique}. Please provide a valid technique."
    technique_details = DEBATE_TECHNIQUES[technique]

    # Read openai node config (exactly matching your YAML)
    try:
        openai_cfg = _read_openai_cfg(config)
    except Exception as e:
        return f"OpenAI config error: {e}"

    model_name = openai_cfg.get("model")
    if not model_name:
        return "openai.llm_model is not configured."

    prompt = generate_prompt(
        user_input,
        all_texts_variable,
        technique,
        technique_details,
        stance,
        position_text
    )
    print("Prompt:", prompt)

    # Build payload; allow max_tokens/temperature from YAML
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
        return f"SSL error when calling API: {e}"
    except requests.exceptions.RequestException as e:
        return f"Network error when calling API: {e}"
    except Exception as e:
        return f"Unknown error when calling API: {e}"


# =========================
# Business: process chunks & select best technique
# =========================
def process_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_texts = []
    skill_scores: Dict[str, float] = {}

    for chunk in chunks:
        all_texts.append(chunk.get("text", ""))
        labels = chunk.get("labels", {}) or {}
        for skill, data in labels.items():
            # Support both english & chinese score fields
            score = data.get("score", data.get("评分", 0))
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

    # Return both EN & CN keys for compatibility
    return {
        # English keys
        "all_texts": all_texts,
        "best_technique": best_skill,
        "top_score": best_score,
        # Chinese-compatible keys (keep to avoid breaking existing code)
        "所有文本": all_texts,
        "最佳辩论技巧": best_skill,
        "最高评分": best_score
    }
