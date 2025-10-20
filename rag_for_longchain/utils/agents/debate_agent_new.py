# -*- coding: utf-8 -*-
"""
Full example: "Resilient JSON Parsing + 3-Round Evaluate–Revise" pipeline
- Generation side: strong prompt to lock JSON
- Parsing side: safe_json_loads with tolerant fixes + fallback
- Main flow: Evaluate -> Optimize (multi-round) -> Final Evaluate
"""
import os
import re
import json
import time
import yaml
import requests
from typing import Dict, Any, Tuple, List, Optional
from requests.exceptions import SSLError, RequestException

# ========================= Resilient JSON Parsing Utils =========================
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
    print(f"[safe_json] Problematic response saved to: {path}")

def safe_json_loads(raw: Any, *, name: str, fallback: Any) -> Any:
    """
    Stepwise fix + parse; on failure, dump raw/attempted text to disk and return fallback.
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

# ========================= Config & Common Utils =========================
CONFIG_FILE = "D:\\conversational_rag/rag_for_longchain\\config\\config.yaml"

def load_config(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error while loading config file: {e}")
        return {}

def get_api_settings(cfg: dict) -> Tuple[str, str, str]:
    """
    Return (api_key, api_base, llm_model)
    Prefer the 'openai' node; root-level fields as fallback.
    """
    ocfg = cfg.get("openai", {})
    api_key = ocfg.get("api_key") or cfg.get("api_key")
    api_base = ocfg.get("api_base") or cfg.get("api_base")
    llm_model = ocfg.get("llm_model") or cfg.get("llm_model")
    if not api_key:
        raise ValueError("API key not configured (openai.api_key or api_key).")
    if not api_base:
        raise ValueError("API base not configured (openai.api_base or api_base).")
    if not llm_model:
        raise ValueError("Model not configured (openai.llm_model or llm_model).")
    return api_key, api_base.rstrip("/"), llm_model

def post_chat_with_ssl_retry(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retry only when SSLZeroReturnError occurs; raise other errors directly.
    """
    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except SSLError as e:
            msg = str(e)
            if "SSLZeroReturnError" in msg:
                print("[Warning] SSLZeroReturnError encountered, retrying...")
                time.sleep(1)
                continue
            raise
        except RequestException:
            raise

def extract_choice_content(data: Dict[str, Any]) -> str:
    """
    Extract choices[0].message.content from OpenAI-style response.
    """
    return data["choices"][0]["message"]["content"]

def clean_json_block(text: str) -> str:
    """
    Remove code fences (``` or ```json) and trim whitespace.
    """
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL).strip()

# ========================= Evaluator =========================
class DebateEvaluator:
    def __init__(self):
        cfg = load_config(CONFIG_FILE)
        self.api_key, self.api_base, self.llm_model = get_api_settings(cfg)
        self.chat_url = f"{self.api_base}/chat/completions"

        self.evaluation_prompt_template = """
[Debate Evaluation Task]
Inputs:
1. Text to evaluate: {counterargument}
2. Our side's advantage points: {my_advantage}
3. Opponent's advantage points: {opponent_advantage}
4. Core points of disagreement: {core_disagreement}
5. Detailed description of debate techniques: {technique_details}

Evaluation criteria:
(1) Argument Handling:
Given the stance of {{counterargument}} (Pro/Con), judge:
- Whether the speech is generated based on its own stance.
- Whether it effectively addresses at least {{min_core}} core points of disagreement.
- Whether it effectively rebuts at least {{min_opponent}} opponent advantages.
- Whether it sufficiently emphasizes at least {{min_my}} of our advantages.

(2) Value System:
- Does it clearly articulate our value advantage?
- Does it present a value hierarchy (instrumental → ultimate)?
- Do the employed techniques match the arguments?

Return JSON in the following format:
{{
    \\"pass_status\\": true,
    \\"detailed_analysis\\": {{
        \\"core_coverage\\": {{
            \\"required\\": {min_core},
            \\"actual\\": 0,
            \\"missing\\": []
        }},
        \\"opponent_rebuttals\\": {{
            \\"required\\": {min_opponent},
            \\"actual\\": 0,
            \\"unaddressed\\": []
        }},
        \\"our_emphasis\\": {{
            \\"required\\": {min_my},
            \\"actual\\": 0,
            \\"missing\\": []
        }},
        \\"value_system\\": {{
            \\"advantage_clarity_score\\": 0.0,
            \\"hierarchy_complete\\": false,
            \\"technique_alignment_score\\": 0.0
        }}
    }},
    \\"revision_suggestions\\": []
}}
You must strictly follow:
1. Output only a single JSON object. No extra explanations, comments, or Markdown fences.
2. All English double quotes inside text must be escaped as \\\\".
3. Newlines in text must be written as \\\\n (double-escaped so \\n is literal).
4. Do not output Markdown, titles, or extra notes—JSON object only.
5. Use English for the output.
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
            my_advantage=";".join(my_advantage),
            opponent_advantage=";".join(opponent_advantage),
            core_disagreement=";".join(core_disagreement),
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
                "pass_status": False,
                "detailed_analysis": {},
                "revision_suggestions": ["Parsing failed: response is not valid JSON"]
            })
            return obj

        except Exception as e:
            print(f"Evaluation request failed: {e}")
            return {
                "pass_status": False,
                "detailed_analysis": {},
                "revision_suggestions": ["Unable to evaluate. Please check inputs or network."],
            }

# ========================= Optimizer =========================
class DebateOptimizer:
    def __init__(self):
        cfg = load_config(CONFIG_FILE)
        self.api_key, self.api_base, self.llm_model = get_api_settings(cfg)
        self.chat_url = f"{self.api_base}/chat/completions"

        self.revision_prompt_template = """
[Debate Revision Task]
Revise the text according to the following evaluation result:
Original text: {counterargument}
Evaluation report: {report}
Our advantage points: {my_advantage}
Opponent advantage points: {opponent_advantage}
Core points of disagreement: {core_disagreement}
Detailed description of techniques: {technique_details}

Revision requirements:
1. Ensure the generated content strictly adheres to its own stance.
2. Address all missing core points and un-rebutted opponent points.
3. Use the techniques suggested in the evaluation.
4. Keep the original style; length change within ±10%.
5. Highlight modifications with ** around changed parts.
6. Output only the debate speech; no other content is allowed.

Return (JSON):
{{
    \\"revised_text\\": \\"string\\",
    \\"techniques_applied\\": [\\"string\\"],
    \\"change_notes\\": \\"string\\"
}}
You must strictly follow:
1. Output only a single JSON object. No extra explanations, comments, or Markdown fences.
2. All English double quotes inside text must be escaped as \\\\".
3. Newlines in text must be written as \\\\n (double-escaped so \\n is literal).
4. Do not output Markdown, titles, or extra notes—JSON object only.
5. Use English for the output.
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
            report=json.dumps(evaluation_report.get("detailed_analysis", {}), ensure_ascii=False),
            my_advantage=";".join(my_advantage),
            opponent_advantage=";".join(opponent_advantage),
            core_disagreement=";".join(core_disagreement),
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

            print("Optimization request abnormal: non-JSON or parsing failed. Keeping original text.")
            return {
                "revised_text": None,  # None indicates no new text this round
                "techniques_applied": [],
                "change_notes": "Optimization failed. Please intervene manually."
            }

        except Exception as e:
            print(f"Optimization request error: {e}")
            return {
                "revised_text": counterargument,
                "techniques_applied": [],
                "change_notes": "Optimization failed. Please intervene manually.",
            }

# ========================= Main Flow (Evaluate–Optimize–Evaluate) =========================
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

        # Optional: force first round to enter optimization
        if i == 0:
            report["pass_status"] = False

        print(f"Round {i + 1} pass_status: {report.get('pass_status')}")

        if report.get("pass_status"):
            break

        revised = optimizer.improve(current_text, report, my_advantage, opponent_advantage, core_disagreement, technique_details)
        print(f"Change notes: {revised.get('change_notes')}")

        # If "revised_text" is None or empty, keep current_text unchanged
        new_text = revised.get("revised_text")
        if isinstance(new_text, str) and new_text.strip():
            current_text = new_text.strip()

        texts.append(current_text)

        time.sleep(1)  # rate limit; avoid hitting quotas

    final_report = evaluator.evaluate(current_text, my_advantage, opponent_advantage, core_disagreement, technique_details)
    print(f"Final pass_status: {final_report.get('pass_status')}")
    return texts, final_report

# ========================= Example Call =========================
if __name__ == "__main__":
    # Initial draft
    counterargument = (
        "First, from a public safety perspective, legalizing cannabis may increase usage rates and raise safety concerns. "
        "Evidence from places such as Colorado in the United States has shown notable rises in violence and traffic incidents associated with cannabis consumption. "
        "This not only harms individuals but also imposes negative externalities on families and communities. "
        "Therefore, to protect citizens and maintain public safety, legalization is an unwise choice."
    )

    # Three elements
    my_advantage = ["Public safety priority", "Youth protection"]
    opponent_advantage = ["Tax revenue", "Reduced law enforcement costs"]
    core_disagreement = ["Trade-off between safety and individual freedom", "Regulatory effectiveness"]

    technique_details = {"Causal argumentation": "Establish a causal chain to justify the conclusion"}

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
