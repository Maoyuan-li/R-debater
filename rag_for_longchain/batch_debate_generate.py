# -*- coding: utf-8 -*-
# @Time    : 2025/5/3 21:36
# @Author  : Maoyuan Li
# @File    : batch_debate_generate.py
# @Software: PyCharm

import re
import json
import traceback
from typing import List, Dict, Any, Tuple
from pathlib import Path

import yaml

# ======== 你的现有模块（保持不变的外部接口）========
from rag_for_longchain.retriever.keyword_retriever import search  # <<< 改为关键词检索
from rag_for_longchain.generator.testgen import (
    generate_counterargument_via_api,
    load_debate_techniques
)
from rag_for_longchain.utils.agents.summarize_agent import ViewpointSummaryAgent
from rag_for_longchain.utils.agents.debate_agent_new import main as debate_main


# ======================= 基础配置 =======================
CONFIG_FILE = r"./config/config.yaml"

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

config = load_config(CONFIG_FILE)


# ======================= 通用工具 =======================
def safe_json_loads(s: str) -> Any:
    """宽松解析 JSON，处理 BOM/截断等常见问题。"""
    try:
        return json.loads(s)
    except Exception:
        pass

    candidate = (s or "").strip().lstrip("\ufeff")
    first = candidate.find("{")
    if first > 0:
        candidate = candidate[first:]
    last_brace = max(candidate.rfind("}"), candidate.rfind("]"))
    if last_brace > 0:
        candidate = candidate[: last_brace + 1]

    return json.loads(candidate)


def _coerce_to_text(x) -> str:
    """把 debate_main 的返回统一收敛为字符串，去掉 **加粗**。"""
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(x, list):
        x = x[-1] if x else ""
    x = x if isinstance(x, str) else str(x)
    x = re.sub(r"\*\*(.*?)\*\*", r"\1", x)
    return x.strip()


def extract_utterances(data: Dict[str, Any]) -> List[str]:
    """仅从 'debate' 列表里提取所有 'utterance'。"""
    out = []
    if isinstance(data, dict) and "debate" in data:
        for entry in data.get("debate", []):
            if isinstance(entry, dict):
                utt = entry.get("utterance", "")
                if isinstance(utt, str) and utt.strip():
                    out.append(utt.strip())
    return out


def recursive_split(text: str, max_chars: int = 600) -> List[str]:
    """
    字符级递归切分，避免额外依赖。
    优先按句号/换行切分，保证 query_list 质量。
    """
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    cut_marks = ["。", "！", "？", "；", "\n", ".", "!", "?", ";"]
    cut_pos = -1
    for pos in range(max_chars, max_chars - 120, -1):
        if pos <= 0 or pos > len(t):
            continue
        if t[pos - 1] in cut_marks:
            cut_pos = pos
            break
    if cut_pos == -1:
        cut_pos = max_chars

    head = t[:cut_pos].strip()
    tail = t[cut_pos:].strip()
    return ([head] if head else []) + recursive_split(tail, max_chars=max_chars)


def process_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    统计 labels 评分，拿到“最佳辩论技巧”和所有文本。
    期望输入：list[{"text": "...", "labels": {...}}, ...]
    """
    all_texts = []
    skill_scores: Dict[str, float] = {}

    for c in (chunks or []):
        txt = c.get("text", "")
        if txt:
            all_texts.append(txt)
        labels = c.get("labels", {}) or {}
        for skill, info in labels.items():
            score = 0.0
            try:
                if isinstance(info, dict):
                    score = float(info.get("评分", info.get("score", 0)) or 0)
                else:
                    score = float(info or 0)
            except Exception:
                score = 0.0
            skill_scores[skill] = skill_scores.get(skill, 0.0) + score

    if skill_scores:
        best_skill = max(skill_scores, key=skill_scores.get)
        best_score = skill_scores[best_skill]
    else:
        best_skill, best_score = None, 0.0

    return {
        "所有文本": [t for t in all_texts if t],
        "最佳辩论技巧": best_skill,
        "最高评分": best_score
    }


# ======================= 三轮优化（基于 summarize_agent + debate_main） =======================
def optimize_three_rounds_via_agents(
    init_text: str,
    technique_details: Dict[str, Any],
    query_list_for_summary: List[str],
    stance: str
) -> Dict[str, str]:
    """
    三轮优化逻辑（完全本地，不走任何 URL）：
      r1: 原稿
      r2: summarize -> debate_main 批判重写
      r3: summarize -> debate_main 再次重写
      final: 取 r3
    """
    current = init_text or ""
    rounds = {"r1": current, "r2": current, "r3": current, "final": current}

    summarizer = ViewpointSummaryAgent()

    for rid in (2, 3):
        try:
            my_adv, opp_adv, core_dis = summarizer.run(query_list_for_summary, current)
            improved = debate_main(current, my_adv, opp_adv, core_dis, technique_details)
            current = _coerce_to_text(improved) or current
        except Exception as e:
            print(f"[优化] 第{rid}轮异常（{stance}）：{e}，沿用上一轮文本。")
        rounds[f"r{rid}"] = current

    rounds["final"] = current
    return rounds


# ======================= 主流程 =======================
DEBATE_TECHNIQUES_FILE = r".\utils\debate_techniques.json"
DEBATE_TECHNIQUES = load_debate_techniques(DEBATE_TECHNIQUES_FILE)


def _read_json_array(path: Path) -> List[Any]:
    """输出文件为 JSON 数组；若不是，备份并重置为空数组。"""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        backup = path.with_suffix(path.suffix + ".bak")
        path.replace(backup)
        print(f"检测到 {path} 非数组格式，已备份为 {backup} 并重新创建数组文件。")
        return []
    except Exception:
        backup = path.with_suffix(path.suffix + ".corrupt.bak")
        try:
            path.replace(backup)
            print(f"检测到 {path} 无法解析为 JSON，已备份为 {backup} 并重置新文件。")
        except Exception:
            print(f"检测到 {path} 无法解析为 JSON，且备份失败，请手动处理。")
        return []


def append_to_json_array(path: Path, item: Dict[str, Any]) -> None:
    arr = _read_json_array(path)
    arr.append(item)
    path.write_text(json.dumps(arr, ensure_ascii=False, indent=4), encoding="utf-8")


def log_error(log_path: Path, file_path: Path, err: Exception) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"出错文件: {str(file_path)}\n")
        f.write("错误类型: " + err.__class__.__name__ + "\n")
        f.write("错误信息: " + str(err) + "\n")
        f.write("堆栈信息:\n")
        f.write(traceback.format_exc())
        f.write("\n")


def process_debate_file(debate_path: Path) -> Dict[str, Any]:
    # 1) 读取输入 JSON（容错）
    raw = debate_path.read_text(encoding="utf-8")
    data = safe_json_loads(raw)

    # 2) 提取用户原始发言并切分 -> 供关键词检索 & 供 summarize agent
    utterances = extract_utterances(data)
    query_list: List[str] = []
    for utt in utterances:
        query_list.extend(recursive_split(utt, max_chars=600))
    if not query_list:
        # 兜底：把整份 JSON 序列化后切分
        query_list = recursive_split(json.dumps(data, ensure_ascii=False), max_chars=600)

    # 3) 关键词检索（你的 keyword_retriever.search）
    #    期望返回：results: [{"text":..., "labels": {...}}, ...], scores: [float, ...]
    best_chunks, _scores = search(query_list)
    retrieve_summary = process_chunks(best_chunks)

    best_skill = retrieve_summary["最佳辩论技巧"]
    technique_details = DEBATE_TECHNIQUES.get(best_skill, {})
    context_joined = "\n\n".join(retrieve_summary["所有文本"])

    # 4) 生成初稿（正/反），即“未优化文本”
    pro_raw = generate_counterargument_via_api(
        data, context_joined, {"最佳辩论技巧": best_skill}, stance="pro"
    )
    con_raw = generate_counterargument_via_api(
        data, context_joined, {"最佳辩论技巧": best_skill}, stance="con"
    )

    # 5) 三轮优化（基于 summarize agent + debate_main）
    pro_rounds = optimize_three_rounds_via_agents(
        init_text=pro_raw,
        technique_details=technique_details,
        query_list_for_summary=query_list,
        stance="pro"
    )
    con_rounds = optimize_three_rounds_via_agents(
        init_text=con_raw,
        technique_details=technique_details,
        query_list_for_summary=query_list,
        stance="con"
    )

    # 6) 最终辩词（若希望再收敛一次，这里再跑一遍）
    my_adv_pro, opp_adv_pro, core_dis_pro = ViewpointSummaryAgent().run(query_list, pro_rounds["final"])
    my_adv_con, opp_adv_con, core_dis_con = ViewpointSummaryAgent().run(query_list, con_rounds["final"])

    pro_out = debate_main(pro_rounds["final"], my_adv_pro, opp_adv_pro, core_dis_pro, technique_details)
    con_out = debate_main(con_rounds["final"], my_adv_con, opp_adv_con, core_dis_con, technique_details)

    final_pro = _coerce_to_text(pro_out)
    final_con = _coerce_to_text(con_out)

    # 7) 只取文件夹名作为 topic
    folder = debate_path.parent.name

    # 8) 组织结果（包含初稿、三轮产物与最终稿）
    result = {
        "topic": folder,
        "model": config.get("model_name", "gpt-5"),
        "best_skill": best_skill,
        "best_skill_score": retrieve_summary["最高评分"],
        # 初稿（便于消融对比）
        "pro_raw": pro_raw,
        "con_raw": con_raw,
        # 三轮过程（便于回溯）
        "pro_r1": pro_rounds["r1"],
        "pro_r2": pro_rounds["r2"],
        "pro_r3": pro_rounds["r3"],
        "con_r1": con_rounds["r1"],
        "con_r2": con_rounds["r2"],
        "con_r3": con_rounds["r3"],
        # 最终稿
        "pro": final_pro,
        "con": final_con
    }
    return result


def main(input_dir: str, output_file: str):
    input_dir_p = Path(input_dir)
    output_path = Path(output_file)
    error_log_path = output_path.with_suffix(output_path.suffix + ".errors.log")

    # 初始化/校验输出文件
    _ = _read_json_array(output_path)

    processed_cnt = 0
    failed_cnt = 0

    for sub in input_dir_p.iterdir():
        if not sub.is_dir():
            continue
        for file in sub.glob("*.json"):
            if file.name == "last_two.json":
                continue
            try:
                result = process_debate_file(file)
                append_to_json_array(output_path, result)
                processed_cnt += 1
                print(f"[OK] 已完成并写入：{file}")
            except Exception as e:
                failed_cnt += 1
                print(f"[ERROR] 处理失败 -> {file}\n原因: {e}")
                log_error(error_log_path, file, e)
            # 每个子目录只取第一场
            break

    print(f"处理完成：成功 {processed_cnt} 场，失败 {failed_cnt} 场。")
    print(f"结果文件：{output_path}")
    if failed_cnt > 0:
        print(f"错误日志：{error_log_path}（包含详细堆栈与文件定位）")


if __name__ == "__main__":
    # ========== 按需修改你的路径 ==========
    input_dir = r".\data\processed_input"
    output_file = r".\data\output\test_output.json"
    main(input_dir, output_file)
