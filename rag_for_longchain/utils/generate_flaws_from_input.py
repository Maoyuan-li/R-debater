# -*- coding: utf-8 -*-
"""
LLM-based Debate Flaw Agent (list-structured input)

适配输入格式（示例）：
[
  {
    "id": "turn_1",
    "side": null,                # "PRO" / "CON" / null
    "utterance": "……",          # 原文陈词；为空则跳过
    "extraction": {              # 将被覆盖写回（若utterance非空）
      "premises": [],
      "assumptions": []
    },
    "flaws_text": "",            # 将被覆盖写回（若utterance非空）
    "note": "空陈词，已跳过"     # 保留原样
  },
  ...
]

输出：
- <input>_flaws.list.json   # 与输入同结构列表，已填充 extraction / flaws_text
- <input>_flaws_report.txt  # 便于人工浏览的报告
"""

import os, re, json, time, yaml, requests, pathlib
from typing import Any, Dict, List, Optional

# ======= 你的路径（保持与你之前一致）=======
CONFIG_PATH = r"D:\converstional_rag\rag_for_longchain\config\config.yaml"
INPUT_FILE  = r"D:\converstional_rag\rag_for_longchain\data\inputdata\复赛第二场香港中文大学vs中山大学_当今中国大陆大麻交易应_不应该合法化.json"
# 如果你要处理你粘贴的“列表输入示例”，把 INPUT_FILE 改成那个文件路径即可

# ======= 你的两段提示词（原样使用）=======
PROMPT_FORMALIZE = """你是一个精确的论证分析助理。
给定一段辩论陈词（中文），请简要提取出该陈词的显式前提（premises）和可能的隐含假设（assumptions）。
要求：
 - 输出有效 JSON（严格的 JSON），包含字段 "premises": [ ... ] 和 "assumptions": [ ... ]（assumptions 可为空数组）。
 - 每个 premise 用一句简短的陈述表示，不要加入外部事实或补充证据。
示例输出：
{"premises": ["A 导致 B", "C 会减少 D"], "assumptions": ["假设 X 成立"]}
"""

PROMPT_FIND_FLAWS = """你是一位辩论逻辑专家。给定一段中文辩论陈词原文与该陈词提取的前提（premises）与隐含假设（assumptions），
请从对手（对方）可能指出的角度，生成**人类可读的**（中文）逻辑漏洞描述（每条 1-3 句），例如：
 - 隐含前提：论者假设 X 成立，但没有证据，缺乏证据会使结论不成立。
 - 因果推断错误：论者把相关性当成因果。
 - 概括过度：从个别案例断定为普遍结论。
每条漏洞后请加一句“修复建议：......”，说明需要何种证据或前提才能弥补该漏洞。
**请仅输出纯文本（非 JSON）**，每行一条，编号即可（1. ... 2. ...）。
"""
# ===== 兼容层：把“外层对象 + debate 列表”转成主流程期望的 List[Record]，无需改 main() =====
import os, json

SIDE_MAP = {
    "PRO": "PRO", "Aff": "PRO", "AFF": "PRO", "正方": "PRO",
    "CON": "CON", "Neg": "CON", "NEG": "CON", "反方": "CON",
    "MIXED": None, None: None
}

def _debate_obj_to_records(data_obj):
    """把 { ..., 'debate': [...] } 转为你主流程期望的 List[Record] 结构"""
    records = []
    debate = data_obj.get("debate") or []
    for i, turn in enumerate(debate, 1):
        stance = (turn.get("stance") or "").strip()
        side = SIDE_MAP.get(stance, None)
        utt  = (turn.get("utterance") or "").strip()
        rec = {
            "id": f"{turn.get('debater', 'turn')}_{i}",
            "side": side,  # "PRO"/"CON"/None
            "utterance": utt,
            "extraction": {"premises": [], "assumptions": []},
            "flaws_text": "",
            "note": "空陈词，已跳过" if not utt else ""
        }
        records.append(rec)
    return records

def _ensure_list_input_and_repoint(path: str) -> str:
    """若输入是对象+debate，则生成 *.prepared.list.json 并返回其路径；否则原样返回。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 已是 List[Record]：直接用
    if isinstance(raw, list):
        return path

    # 是对象 + debate 列表：转换并重定向
    if isinstance(raw, dict) and isinstance(raw.get("debate"), list):
        records = _debate_obj_to_records(raw)
        new_path = os.path.splitext(path)[0] + ".prepared.list.json"
        with open(new_path, "w", encoding="utf-8") as wf:
            json.dump(records, wf, ensure_ascii=False, indent=2)
        return new_path

    # 其它结构：让主流程按你原有逻辑报错
    return path

# ---- 关键：重定向全局 INPUT_FILE 到“已转换的列表文件” ----
INPUT_FILE = _ensure_list_input_and_repoint(INPUT_FILE)

# ======= 小工具 =======

def load_yaml_config(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "openai" not in cfg:
        raise RuntimeError(f"配置缺少 'openai' 节点：{path}")
    o = cfg["openai"]
    api_key = o.get("api_key")
    api_base = (o.get("api_base") or "").rstrip("/")
    llm_model = o.get("llm_model")
    if not api_key:   raise RuntimeError("config.yaml 缺少 openai.api_key")
    if not api_base:  raise RuntimeError("config.yaml 缺少 openai.api_base")
    if not llm_model: raise RuntimeError("config.yaml 缺少 openai.llm_model")
    return {"api_key": api_key, "api_base": api_base, "llm_model": llm_model}

def call_chat(api_base: str, api_key: str, model: str, messages: List[Dict[str,str]],
              temperature=0.2, max_tokens=1024, retries=3, timeout=60) -> str:
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json; charset=utf-8"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    err = None
    for i in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            err = RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        except Exception as e:
            err = e
        time.sleep(1.2*(i+1))
    raise err

def safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # ```json ... ```
    import re as _re
    for blk in _re.findall(r"```(?:json)?\s*([\s\S]*?)```", s, flags=_re.I):
        try:
            return json.loads(blk.strip())
        except Exception:
            continue
    # 首个大括号块
    m = _re.search(r"(\{(?:[^{}]|(?1))*\})", s, flags=_re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

def ensure_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def invert_side(side: Optional[str]) -> Optional[str]:
    return {"PRO":"CON", "CON":"PRO"}.get((side or "").upper())

def run_formalize(api_base, api_key, model, utter):
    sys_p = "你是一个精确的论证分析助理。"
    usr_p = f"{PROMPT_FORMALIZE}\n\n=== 辩论陈词原文 ===\n{utter}\n"
    out = call_chat(api_base, api_key, model,
                    [{"role":"system","content":sys_p},{"role":"user","content":usr_p}],
                    temperature=0.1, max_tokens=800)
    parsed = safe_json_parse(out) or {"premises":[], "assumptions":[]}
    parsed["premises"]   = [str(x).strip() for x in ensure_list(parsed.get("premises")) if str(x).strip()]
    parsed["assumptions"]= [str(x).strip() for x in ensure_list(parsed.get("assumptions")) if str(x).strip()]
    return parsed

def run_find_flaws(api_base, api_key, model, utter, extraction, side):
    opp = invert_side(side) or "对方"
    sys_p = "你是一位辩论逻辑专家。"
    usr_p = (
        f"{PROMPT_FIND_FLAWS}\n\n"
        f"【说明】当前发言持方：{side or '未知'}；请从『{opp}』视角指出问题。\n"
        f"=== 辩论陈词原文 ===\n{utter}\n\n"
        f"=== 提取的 premises/assumptions（JSON）===\n{json.dumps(extraction, ensure_ascii=False)}\n"
        f"=== 输出格式提醒 ===\n仅输出纯文本，每行一条，形如：\n"
        f"1. （漏洞类型）问题描述…… 修复建议：……\n"
    )
    out = call_chat(api_base, api_key, model,
                    [{"role":"system","content":sys_p},{"role":"user","content":usr_p}],
                    temperature=0.3, max_tokens=800)
    # 清理为行
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip() and not ln.strip().startswith("```")]
    return "\n".join(lines)

def make_out_paths(input_path: str):
    p = pathlib.Path(input_path)
    base = p.with_suffix("")
    return str(base) + "_flaws.list.json", str(base) + "_flaws_report.txt"

# ======= 主流程（适配“列表结构输入”）=======
def main():
    cfg = load_yaml_config(CONFIG_PATH)
    api_key, api_base, model = cfg["api_key"], cfg["api_base"], cfg["llm_model"]

    # 读取列表
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("期望输入为 List[Record]；请确认你的文件最外层是数组（[...]）。")

    out_list: List[Dict[str, Any]] = []
    report_lines: List[str] = []

    for idx, rec in enumerate(data, 1):
        # 原样复制，确保未处理项也不丢字段
        rec_out = dict(rec)
        utter = (rec.get("utterance") or "").strip()
        side  = rec.get("side")  # 允许 None / "PRO" / "CON"

        if not utter:
            # 空陈词：保持原样（不覆盖原 extraction / flaws_text / note）
            out_list.append(rec_out)
            # 报告也标注一下
            report_lines += [
                f"===== 段落 {idx} | ID={rec.get('id','?')} | 持方={side or '未知'} =====",
                "【空陈词】已跳过\n"
            ]
            continue

        # 有文本：跑两段提示词
        try:
            extraction = run_formalize(api_base, api_key, model, utter)
        except Exception as e:
            extraction = {"premises": [], "assumptions": []}
            report_lines.append(f"[WARN] formalize 失败（{rec.get('id','?')}）：{e}")

        try:
            flaws_text = run_find_flaws(api_base, api_key, model, utter, extraction, side)
        except Exception as e:
            flaws_text = f"生成失败：{e}"
            report_lines.append(f"[WARN] find_flaws 失败（{rec.get('id','?')}）：{e}")

        # 写回当前记录（仅覆盖这两个字段）
        rec_out["extraction"] = extraction
        rec_out["flaws_text"] = flaws_text
        out_list.append(rec_out)

        # 报告
        report_lines += [
            f"===== 段落 {idx} | ID={rec.get('id','?')} | 持方={side or '未知'} =====",
            "【陈词原文】",
            utter,
            "",
            "【抽取的前提 / 隐含假设】",
            json.dumps(extraction, ensure_ascii=False, indent=2),
            "",
            "【对手可能指出的逻辑漏洞（含修复建议）】",
            flaws_text if flaws_text else "(无)",
            "\n"
        ]

        time.sleep(0.4)  # 轻微节流

    # 写文件
    out_json, out_txt = make_out_paths(INPUT_FILE)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("完成 ✅")
    print(f"- 列表结构输出：{out_json}")
    print(f"- 文本报告：    {out_txt}")

if __name__ == "__main__":
    main()
