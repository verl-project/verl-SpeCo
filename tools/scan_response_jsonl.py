"""Scan conversation JSONL containing user prompts and assistant responses.

Edit the configuration below, then run: python tools/scan_response_jsonl.py
This performs basic structural checks without modifying the input file.
"""

import json
from collections import Counter, defaultdict

# ================= 配置 =================
DATASET_PATH = "/path/to/data.jsonl"
MAX_RECORDS = 0  # 前多少条非空记录；0 = 全部
EXAMPLES_PER_TYPE = 5  # 每类打印几个原文件行号
# ========================================


def main():
    if MAX_RECORDS < 0 or EXAMPLES_PER_TYPE < 0:
        raise ValueError("MAX_RECORDS 和 EXAMPLES_PER_TYPE 不能为负数")

    counts = Counter()
    examples = defaultdict(list)
    scanned = 0

    def record(category, line_number):
        counts[category] += 1
        if len(examples[category]) < EXAMPLES_PER_TYPE:
            examples[category].append(line_number)

    def check_row(row, line_number):
        if isinstance(row, list):
            turns = row
        elif isinstance(row, dict):
            turns = next(
                (
                    row[key]
                    for key in ("conversations", "conversation", "messages")
                    if isinstance(row.get(key), list)
                ),
                None,
            )
        else:
            turns = None

        if turns is None:
            record("missing_conversation_list", line_number)
            return
        if not turns:
            record("empty_conversation", line_number)
            return

        roles = []
        has_bad_message = False
        has_bad_content = False
        has_blank_content = False
        for turn in turns:
            if not isinstance(turn, dict):
                has_bad_message = True
                continue
            role = turn.get("from", turn.get("role"))
            roles.append(role.strip().lower() if isinstance(role, str) else "")
            # 只有 value 字段缺失时才回退到 content；显式的 null 不回退。
            content = turn.get("value", turn.get("content"))
            if not isinstance(content, str) or content == "":
                has_bad_content = True
            elif not content.strip():
                has_blank_content = True

        has_human = any(role in {"human", "user"} for role in roles)
        has_assistant = any(role in {"assistant", "gpt"} for role in roles)
        problems = {
            "missing_human": not has_human,
            "missing_assistant": not has_assistant,
            "non_object_message": has_bad_message,
            "invalid_or_empty_content": has_bad_content,
            "whitespace_only_content": has_blank_content,
        }
        for category, present in problems.items():
            if present:
                record(category, line_number)
        if not any(problems.values()):
            record("passes_basic_checks", line_number)

    with open(DATASET_PATH, "r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if MAX_RECORDS and scanned >= MAX_RECORDS:
                break
            if not line.strip():
                record("blank_line", line_number)
                continue
            scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                record("invalid_json", line_number)
            else:
                check_row(row, line_number)
            if scanned % 100000 == 0:
                print(f"已扫描 {scanned:,} 条", flush=True)

    labels = {
        "missing_human": "缺少 human/user",
        "missing_assistant": "缺少 assistant/gpt",
        "invalid_or_empty_content": "value/content 缺失、非字符串或空字符串",
        "whitespace_only_content": "value/content 只有空白字符",
        "non_object_message": "消息不是 JSON 对象",
        "missing_conversation_list": "没有有效的对话列表字段",
        "empty_conversation": "对话列表为空",
        "invalid_json": "JSON 解析失败",
        "passes_basic_checks": "通过上述基础检查",
        "blank_line": "空白行",
    }
    print(f"\n扫描文件：{DATASET_PATH}")
    print(f"扫描非空记录数：{scanned:,}")
    print("各类别可能重叠；通过基础检查不代表通过全部训练校验。\n")
    for category, label in labels.items():
        n = counts[category]
        ratio = f"{n / scanned:.2%}" if scanned and category != "blank_line" else "-"
        print(f"{label}: {n:,}  ({ratio})")
        if examples[category]:
            print(f"  示例原文件行号：{examples[category]}")


if __name__ == "__main__":
    main()
