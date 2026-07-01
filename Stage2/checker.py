import json
import re
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


SAFETY_PATTERNS = {
    "death_prediction": [
        r"\byou will die\b", r"\bdeath is certain\b", r"\bmar jaoge\b", r"\bmrit(yu|yu)\b"
    ],
    "illness_prediction": [
        r"\byou have cancer\b", r"\byou have heart disease\b", r"\bbimari pakki\b",
        r"\bkundli.*(cancer|heart attack|serious illness)"
    ],
    "guarantee": [
        r"\bguarantee\b", r"\b100%\b", r"\bdefinitely\b", r"\bpakka\b",
        r"\bwill surely\b", r"\bzaroor hoga\b"
    ],
    "fear_selling": [
        r"\bpay\b.*\bpuja\b", r"\b51000\b", r"\bvarna\b.*\bbarbaad\b",
        r"\botherwise.*ruined\b", r"\bmust do.*remedy\b"
    ],
    "financial_promise": [
        r"\bbecome rich\b", r"\bcrorepati\b", r"\bmoney will come\b",
        r"\bstock.*sure\b", r"\bprofit guaranteed\b"
    ],
}


def read_jsonl(path):
    chats = []
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chats.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_no}: Invalid JSON - {e}")

    return chats, errors


def chat_text(chat):
    return " ".join(
        msg.get("content", "") for msg in chat.get("messages", [])
    )


def validate_structure(chat):
    issues = []

    if not isinstance(chat, dict):
        return ["Chat is not a JSON object"]

    if "messages" not in chat or not isinstance(chat["messages"], list):
        return ["Missing or invalid messages list"]

    messages = chat["messages"]

    if len(messages) < 3:
        issues.append("Chat has fewer than 3 messages")

    if not messages or messages[0].get("role") != "system":
        issues.append("First message must be system")

    expected = "user"
    for i, msg in enumerate(messages[1:], start=1):
        role = msg.get("role")

        if role != expected:
            issues.append(
                f"Message {i} should be {expected}, found {role}"
            )

        if "content" not in msg or not msg["content"].strip():
            issues.append(f"Message {i} has empty content")

        expected = "assistant" if expected == "user" else "user"

    return issues


def word_count(chat):
    return len(re.findall(r"\w+", chat_text(chat)))


def safety_flags(chat):
    text = chat_text(chat).lower()
    flags = []

    for category, patterns in SAFETY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                flags.append(category)
                break

    return flags


def find_duplicates(chats, threshold=0.85):
    texts = [chat_text(chat) for chat in chats]

    if len(texts) < 2:
        return []

    vectorizer = TfidfVectorizer().fit_transform(texts)
    sim_matrix = cosine_similarity(vectorizer)

    duplicates = []

    for i in range(len(chats)):
        for j in range(i + 1, len(chats)):
            if sim_matrix[i][j] >= threshold:
                duplicates.append({
                    "chat_1": chats[i].get("id", i),
                    "chat_2": chats[j].get("id", j),
                    "similarity": round(float(sim_matrix[i][j]), 3)
                })

    return duplicates


def run_checker(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chats, json_errors = read_jsonl(input_path)

    report = []
    valid_chats = []

    report.append("VEDAZ CHAT CHECKER REPORT")
    report.append("=" * 35)
    report.append(f"Input file: {input_path}")
    report.append(f"Total chats read: {len(chats)}")
    report.append(f"JSON errors: {len(json_errors)}")
    report.append("")

    for err in json_errors:
        report.append(err)

    report.append("\nCHAT LEVEL RESULTS")
    report.append("-" * 35)

    for idx, chat in enumerate(chats):
        chat_id = chat.get("id", f"chat_{idx}")
        structure_issues = validate_structure(chat)
        flags = safety_flags(chat)
        wc = word_count(chat)

        report.append(f"\nChat ID: {chat_id}")
        report.append(f"Word count: {wc}")

        if structure_issues:
            report.append("Structure issues:")
            for issue in structure_issues:
                report.append(f"  - {issue}")
        else:
            report.append("Structure: OK")

        if flags:
            report.append(f"Safety flags: {', '.join(flags)}")
        else:
            report.append("Safety: OK")

        if not structure_issues and not flags:
            valid_chats.append(chat)

    duplicates = find_duplicates(valid_chats)

    report.append("\nDUPLICATE / NEAR-DUPLICATE CHECK")
    report.append("-" * 35)

    if duplicates:
        for dup in duplicates:
            report.append(str(dup))
    else:
        report.append("No near-duplicates found.")

    if len(valid_chats) >= 2:
        train, test = train_test_split(
            valid_chats,
            test_size=0.2,
            random_state=42
        )
    else:
        train, test = valid_chats, []

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    report_path = output_dir / "checker_report.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        for chat in train:
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for chat in test:
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")

    report.append("\nSPLIT SUMMARY")
    report.append("-" * 35)
    report.append(f"Valid chats: {len(valid_chats)}")
    report.append(f"Training chats: {len(train)}")
    report.append(f"Test chats: {len(test)}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/vedaz_astrologer_finetune.jsonl")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    run_checker(args.input, args.output)