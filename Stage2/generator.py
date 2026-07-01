import json
import os
import time
from pathlib import Path

from google import genai
from checker import validate_structure, safety_flags


MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """
You are creating safe training data for Vedaz AI Astrologer.

Create ONE complete chat in valid JSON format with:
- id
- tags
- messages

Rules:
- First message must be system.
- Then alternate user and assistant turns.
- Assistant must be warm, honest, non-fatalistic.
- Never predict death, disease, or guaranteed misfortune.
- Never promise money, marriage, job, health, or legal result.
- Health issues must be redirected to doctors.
- Legal or major financial matters must be redirected to professionals.
- Remedies must be supportive spiritual practices, never guarantees.
- Use the user's language/register.

Return ONLY JSON. No markdown. No explanation.
"""


TOPICS = [
    "career delay, Hindi",
    "marriage compatibility, Hinglish",
    "health anxiety, Hindi, safe doctor redirect",
    "business loan decision, Hinglish, financial safety",
    "student exam fear, Hindi",
    "skeptical user, Hinglish",
    "Kaal Sarp fear-selling, Hindi",
    "relationship loyalty anxiety, Hinglish",
    "gemstone for money, Hindi",
    "foreign travel delay, English",
]


def extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")

    return json.loads(text[start:end + 1])


def make_prompt(topic: str, index: int) -> str:
    return f"""
Topic: {topic}

Create a Vedaz training chat.

Use this id exactly:
generated_{index:03d}

Return JSON in this shape:
{{
  "id": "generated_{index:03d}",
  "tags": ["tag1", "tag2"],
  "messages": [
    {{
      "role": "system",
      "content": "..."
    }},
    {{
      "role": "user",
      "content": "..."
    }},
    {{
      "role": "assistant",
      "content": "..."
    }}
  ]
}}
"""


def generate_one_chat(client, topic: str, index: int) -> dict:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=SYSTEM_INSTRUCTION + "\n\n" + make_prompt(topic, index),
    )

    if not response.text:
        raise ValueError("Empty model response.")

    return extract_json(response.text)


def is_valid_generated_chat(chat: dict) -> bool:
    structure_issues = validate_structure(chat)
    flags = safety_flags(chat)

    if structure_issues:
        print("Rejected: structure issue", structure_issues)
        return False

    if flags:
        print("Rejected: safety flags", flags)
        return False

    return True


def main():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)

    output_path = Path("data/generated_chats.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for index, topic in enumerate(TOPICS, start=1):
            print(f"\nGenerating chat {index}: {topic}")

            try:
                chat = generate_one_chat(client, topic, index)

                if is_valid_generated_chat(chat):
                    f.write(json.dumps(chat, ensure_ascii=False) + "\n")
                    saved += 1
                    print("Saved ✅")
                else:
                    print("Skipped ❌")

                time.sleep(2)

            except Exception as e:
                print(f"Failed: {e}")

    print(f"\nDone. Total good chats saved: {saved}")
    print(f"Saved file: {output_path}")


if __name__ == "__main__":
    main()