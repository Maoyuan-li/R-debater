# -*- coding: utf-8 -*-
"""
完整示例：带“抗脏 JSON 解析 + 三轮评估-优化”流水线
- 生成端：强提示锁定 JSON
- 解析端：safe_json_loads 容错修复 + 失败兜底
- 主流程：评估 -> 优化（可多轮）-> 最终评估
"""
import os
import re
import json
import time
import yaml
import requests
from typing import Dict, Any, Tuple, List, Optional
from requests.exceptions import SSLError, RequestException

# ========================= 抗脏 JSON 解析工具 =========================
_BAD_DIR = "bad_json"
os.makedirs(_BAD_DIR, exist_ok=True)

_SMART_QUOTES = {"“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "’": "'", "‘": "'"}

def _sj_strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _sj_fix_smart_quotes(s: str) -> str:
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    return s

def _sj_remove_comments(s: str) -> str:
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return s

def _sj_remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def _sj_maybe_quote_keys(s: str) -> str:
    pattern = r'(?P<prefix>[\{\s,])(?P<key>[A-Za-z_][A-Za-z0-9_\-\.]*)\s*:(?P<after>\s*)'
    repl = r'\g<prefix>"\g<key>":\g<after>'
    return re.sub(pattern, repl, s)

def _sj_single_to_double_quotes(s: str) -> str:
    s = re.sub(r"(?P<prefix>[:\s\{\[,])\s*'([^'\\]*(?:\\.[^'\\]*)*)'", r'\g<prefix> "\2"', s)
    s = re.sub(r"'(?P<key>[A-Za-z_][A-Za-z0-9_\-\.]*)'\s*:", r'"\g<key>":', s)
    return s

def _sj_extract_first_balanced_json(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def _sj_dump_bad(name: str, raw: str, fixed: Optional[str] = None) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_BAD_DIR, f"{ts}_{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== RAW ===\n")
        f.write(raw if isinstance(raw, str) else str(raw))
        f.write("\n\n=== FIXED ===\n")
        f.write("" if fixed is None else fixed)
    print(f"[safe_json] 已保存问题响应到: {path}")

def safe_json_loads(raw: Any, *, name: str, fallback: Any) -> Any:
    """
    逐步修复 + 解析；失败则将原文/尝试版本落盘，并返回 fallback。
    """
    if not isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            pass

    candidates = []
    s = _sj_strip_code_fences(str(raw))
    candidates.append(s)

    s1 = _sj_remove_comments(_sj_fix_smart_quotes(s))
    s1 = _sj_remove_trailing_commas(s1)
    candidates.append(s1)

    s2 = _sj_maybe_quote_keys(_sj_single_to_double_quotes(s1))
    s2 = _sj_remove_trailing_commas(s2)
    candidates.append(s2)

    s3 = _sj_extract_first_balanced_json(s2)
    if s3:
        candidates.append(s3)

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue

    _sj_dump_bad(name, str(raw), candidates[-1] if candidates else None)
    return fallback

# ========================= 配置与常用工具 =========================
CONFIG_FILE = "D:/converstional_rag/rag_for_longchain/config/config.yaml"

def load_config(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"加载配置文件时发生错误: {e}")
        return {}

def get_api_settings(cfg: dict) -> Tuple[str, str, str]:
    """
    返回 (api_key, api_base, llm_model)
    优先从 openai 节点取；根级字段作为兜底。
    """
    ocfg = cfg.get("openai", {})
    api_key = ocfg.get("api_key") or cfg.get("api_key")
    api_base = ocfg.get("api_base") or cfg.get("api_base")
    llm_model = ocfg.get("llm_model") or cfg.get("llm_model")
    if not api_key:
        raise ValueError("API密钥未配置（openai.api_key 或 api_key）。")
    if not api_base:
        raise ValueError("API基础地址未配置（openai.api_base 或 api_base）。")
    if not llm_model:
        raise ValueError("模型未配置（openai.llm_model 或 llm_model）。")
    return api_key, api_base.rstrip("/"), llm_model

def post_chat_with_ssl_retry(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    仅在出现 SSLZeroReturnError 时重试；其它错误直接抛出。
    """
    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except SSLError as e:
            msg = str(e)
            if "SSLZeroReturnError" in msg:
                print("[警告] 遇到 SSLZeroReturnError，正在重试...")
                time.sleep(1)
                continue
            raise
        except RequestException:
            raise

def extract_choice_content(data: Dict[str, Any]) -> str:
    """
    兼容 OpenAI 风格返回，提取 choices[0].message.content。
    """
    return data["choices"][0]["message"]["content"]

def clean_json_block(text: str) -> str:
    """
    去掉围栏（``` 或 ```json）并裁剪空白。
    """
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL).strip()

# ========================= 评估器 =========================
class DebateEvaluator:
    def __init__(self):
        cfg = load_config(CONFIG_FILE)
        self.api_key, self.api_base, self.llm_model = get_api_settings(cfg)
        self.chat_url = f"{self.api_base}/chat/completions"

        self.evaluation_prompt_template = """
[辩论评估任务]
输入要素：
1. 待评估文本：{counterargument}
2. 正方优势论点：{my_advantage}
3. 反方优势论点：{opponent_advantage}
4. 核心分析点：{core_disagreement}
5. 辩论技巧的详细描述：{technique_details}

评估标准：
(1) 论点处理：
根据{counterargument}的持方（正方/反方），判断：
- 这段辩论陈词是否基于自己的持方去生成
- 是否有效反驳核心分析点？（至少覆盖{min_core}个）
- 是否有效反驳对方至少{min_opponent}个优势论点？
- 是否充分强调我方至少{min_my}个优势论点？

(2) 价值体系：
- 是否明确阐述我方价值优势？
- 是否完成价值排序（工具价值→终极价值）？
- 辩论技巧是否与论点匹配？

请按JSON格式返回：
{{
    "达标状态": true,
    "详细分析": {{
        "核心分析点覆盖": {{
            "应覆盖数": {min_core},
            "实际覆盖数": 0,
            "缺失列表": []
        }},
        "对方论点反驳": {{
            "应反驳数": {min_opponent},
            "实际反驳数": 0,
            "未处理列表": []
        }},
        "我方论点强调": {{
            "应强调数": {min_my},
            "实际强调数": 0,
            "缺失列表": []
        }},
        "价值体系评估": {{
            "优势阐述评分": 0.0,
            "层次完整性": false,
            "技巧匹配度": 0.0
        }}
    }},
    "修改建议": []
}}
你必须严格遵循以下要求生成输出：
1. 只输出一个 JSON 对象，不要输出任何额外的解释、注释或 Markdown 代码块标记。
2. 文本里的所有英文双引号必须写成 \\"。
3. 文本里的换行必须写成 \\\\n（注意双重转义，保证落地时是 \\n 字面量）。
4. 禁止输出 Markdown、标题、额外说明，只能返回 JSON 对象本身。
""".strip()

    def evaluate(
        self,
        counterargument: str,
        my_advantage: List[str],
        opponent_advantage: List[str],
        core_disagreement: List[str],
        technique_details: Any,
    ) -> Dict[str, Any]:
        technique_str = technique_details if isinstance(technique_details, str) \
            else json.dumps(technique_details, ensure_ascii=False)

        prompt = self.evaluation_prompt_template.format(
            counterargument=counterargument,
            my_advantage="；".join(my_advantage),
            opponent_advantage="；".join(opponent_advantage),
            core_disagreement="；".join(core_disagreement),
            technique_details=technique_str,
            min_core=min(2, len(core_disagreement)),
            min_opponent=min(1, len(opponent_advantage)),
            min_my=min(2, len(my_advantage)),
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.llm_model,
            "stream": False,
            "temperature": 0.2,
            "top_p": 1.0,
            "messages": [
                {"role": "system", "content": "You are a JSON-only generator. Only output a single valid JSON object."},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            data = post_chat_with_ssl_retry(self.chat_url, headers, payload)
            content = extract_choice_content(data)
            clean = clean_json_block(content)
            obj = safe_json_loads(clean, name="debate_evaluate", fallback={
                "达标状态": False,
                "详细分析": {},
                "修改建议": ["解析失败：返回内容不是合法JSON"]
            })
            return obj

        except Exception as e:
            print(f"评估请求失败: {e}")
            return {
                "达标状态": False,
                "详细分析": {},
                "修改建议": ["无法评估，请检查输入或网络状况"],
            }

# ========================= 优化器 =========================
class DebateOptimizer:
    def __init__(self):
        cfg = load_config(CONFIG_FILE)
        self.api_key, self.api_base, self.llm_model = get_api_settings(cfg)
        self.chat_url = f"{self.api_base}/chat/completions"

        self.revision_prompt_template = """
[辩论优化任务]
根据以下评估结果进行文本修改：
原始文本：{counterargument}
评估报告：{report}
我方优势论点：{my_advantage}
对方优势论点：{opponent_advantage}
核心分析点：{core_disagreement}
辩论技巧的详细描述：{technique_details}

优化要求：
1. 必须确认生成的内容严格遵守自己所持有的持方
2. 必须处理所有缺失的核心分析点和未反驳论点
3. 使用评估建议中的辩论技巧
4. 保持原文风格，字数增减不超过10%
5. 突出显示修改部分（用**包围）
6. 只可以输出辩论陈词，严禁输出除辩论陈词外的其它内容

请返回（JSON）：
{{
    "修正文本": "string",
    "应用的技巧": ["string"],
    "修改说明": "string"
}}
注意：
你必须严格遵循以下要求生成输出：
1. 只输出一个 JSON 对象，不要输出任何额外的解释、注释或 Markdown 代码块标记。
2. 文本里的所有英文双引号必须写成 \\"。
3. 文本里的换行必须写成 \\\\n（注意双重转义，保证落地时是 \\n 字面量）。
4. 禁止输出 Markdown、标题、额外说明，只能返回 JSON 对象本身。
""".strip()

    def improve(
        self,
        counterargument: str,
        evaluation_report: Dict[str, Any],
        my_advantage: List[str],
        opponent_advantage: List[str],
        core_disagreement: List[str],
        technique_details: Any,
    ) -> Dict[str, Any]:

        technique_str = technique_details if isinstance(technique_details, str) \
            else json.dumps(technique_details, ensure_ascii=False)

        prompt = self.revision_prompt_template.format(
            counterargument=counterargument,
            report=json.dumps(evaluation_report.get("详细分析", {}), ensure_ascii=False),
            my_advantage="；".join(my_advantage),
            opponent_advantage="；".join(opponent_advantage),
            core_disagreement="；".join(core_disagreement),
            technique_details=technique_str,
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.llm_model,
            "stream": False,
            "temperature": 0.2,
            "top_p": 1.0,
            "messages": [
                {"role": "system", "content": "You are a JSON-only generator. Only output a single valid JSON object."},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            data = post_chat_with_ssl_retry(self.chat_url, headers, payload)
            content = extract_choice_content(data)
            clean = clean_json_block(content)
            obj = safe_json_loads(clean, name="debate_optimize", fallback=None)
            if isinstance(obj, dict):
                return obj

            print("优化请求异常：返回非JSON或解析失败，已保留原文本继续。")
            return {
                "修正文本": None,  # None 表示本轮未拿到新文本
                "应用的技巧": [],
                "修改说明": "优化请求失败，请人工介入。"
            }

        except Exception as e:
            print(f"优化请求异常: {e}")
            return {
                "修正文本": counterargument,
                "应用的技巧": [],
                "修改说明": "优化请求失败，请人工介入。",
            }

# ========================= 主流程（评估-优化-评估） =========================
def main(counterargument: str,
         my_advantage: List[str],
         opponent_advantage: List[str],
         core_disagreement: List[str],
         technique_details: Any,
         max_iterations: int = 3):
    evaluator = DebateEvaluator()
    optimizer = DebateOptimizer()

    current_text = counterargument
    texts = [current_text]

    for i in range(max_iterations):
        report = evaluator.evaluate(current_text, my_advantage, opponent_advantage, core_disagreement, technique_details)

        # 可选：如果你希望第一轮一定进入优化，可强制置 False
        if i == 0:
            report["达标状态"] = False

        print(f"第{i + 1}轮评估结果: {report.get('达标状态')}")

        if report.get("达标状态"):
            break

        revised = optimizer.improve(current_text, report, my_advantage, opponent_advantage, core_disagreement, technique_details)
        print(f"修改说明: {revised.get('修改说明')}")

        # 若“修正文本”为 None 或空，则保持 current_text 不变
        new_text = revised.get("修正文本")
        if isinstance(new_text, str) and new_text.strip():
            current_text = new_text.strip()

        texts.append(current_text)

        time.sleep(1)  # 限速，避免命中速率限制

    final_report = evaluator.evaluate(current_text, my_advantage, opponent_advantage, core_disagreement, technique_details)
    print(f"最终达标状态: {final_report.get('达标状态')}")
    return texts, final_report

# ========================= 示例调用 =========================
if __name__ == "__main__":
    # 初稿
    counterargument = (
        "首先，从公共安全的角度来看，合法化大麻可能会导致更高的吸食率和社会安全问题。研究表明，"
        "在允许大麻合法化的地区，比如美国科罗拉多州，因吸食大麻导致的暴力犯罪和交通事故的发生率显著上升。"
        "这不仅影响个人的安全，也对家庭和社会造成了负面影响。因此，我方认为，出于保护民众和维护公共安全的目的，"
        "合法化大麻是不明智的选择。"
    )

    # 三要素
    my_advantage = ["公共安全优先", "青少年保护"]
    opponent_advantage = ["税收收益", "执法成本下降"]
    core_disagreement = ["公共安全与自由选择的权衡", "监管有效性"]

    technique_details = {"因果论证": "建立因果链条证明结论的合理性"}

    final_texts, final_report = main(
        counterargument,
        my_advantage,
        opponent_advantage,
        core_disagreement,
        technique_details,
        max_iterations=3
    )

    print("debug_final_texts:", final_texts)
    print("debug_final_report:", json.dumps(final_report, ensure_ascii=False, indent=2))
