#!/usr/bin/env python3
"""【計測用】SO-101の把持確認10枚を、Jevと同じ質問・選択肢でGeminiに判定させ直す。

元の記録（observations.json の gemini）は、日本語の自由記述で「成功/失敗」と根拠を書かせていた。
Jev は 3 択の Choice（GRIP_INSTRUCTIONS / GRIP_CRITERIA、制御コード 6979a11 の src/jev/client.py）なので、
質問文と選択肢の文言を Jev と一字一句そろえ、答えは 3 択の JSON だけを返させる。
温度 0.5・thinking 既定は、元の制御コードの Gemini 呼び出しと同じ。

  uv run --env-file <GEMINI_API_KEY を含む .env> --python 3.12 --with google-genai scripts/so101-gemini-aligned.py
出力: experiments/recipes/so101/gemini-aligned.json
"""
import json
import os
import statistics as st
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

DIR = Path(__file__).resolve().parents[1] / "experiments/recipes/so101"
MODEL = "gemini-robotics-er-2-preview"
TEMPERATURE = 0.5
# 制御コード src/gemini/client.py と同じ単価（1M トークンあたり USD）
PRICE_INPUT, PRICE_OUTPUT = 1.0, 5.0

# 制御コード 6979a11 の src/jev/client.py から一字一句写す
GRIP_CRITERIA = {
    "success": "The gripper jaws are holding `target`.",
    "failure": "The gripper jaws are empty, or they are holding an object other than `target`.",
    "cannot_tell": (
        "The image does not show enough to decide, e.g. the jaws are out of frame, "
        "hidden, or too blurred to see."
    ),
}
GRIP_INSTRUCTIONS = (
    "`photo` is a close-up from a camera mounted near a robot arm's gripper, "
    "taken right after the arm tried to pick up `target` and lift it. "
    "Judging only from what the image shows, is the gripper holding `target`?"
)


def prompt(target):
    # Jev の state（target と photo）と質問・選択肢を、同じ文言のまま 1 つの文にする。photo は添付画像を指す
    state = json.dumps({"target": target, "photo": "(the attached image)"}, ensure_ascii=False)
    choices = "\n".join(f"- {key}: {text}" for key, text in GRIP_CRITERIA.items())
    return f"state: {state}\n\nQuestion: {GRIP_INSTRUCTIONS}\n\nChoices:\n{choices}"


def main():
    observations = json.loads((DIR / "observations.json").read_text())
    target = observations["protocol"]["fixed_pick_action"]["label"]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema={"type": "OBJECT", "properties": {"outcome": {"type": "STRING", "enum": list(GRIP_CRITERIA)}},
                         "required": ["outcome"]},
    )
    rows = []
    for case in observations["cases"]:
        image = (DIR / case["image"]).read_bytes()
        mime = "image/png" if case["image"].endswith(".png") else "image/jpeg"
        started = time.perf_counter()
        response = client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=image, mime_type=mime), prompt(target)],
            config=config,
        )
        latency = time.perf_counter() - started
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        outcome = json.loads(response.text)["outcome"]
        rows.append({
            "id": case["id"], "label": case["provisional_visual_label"], "outcome": outcome,
            "model_version": response.model_version, "latency_s": round(latency, 3),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "thinking_tokens": usage.thoughts_token_count or 0,
            "cost_usd": (input_tokens * PRICE_INPUT + output_tokens * PRICE_OUTPUT) / 1e6,
            "jev_verdict": case["jev"]["verdict"], "jev_latency_s": case["jev"]["latency_s"],
        })
        print(rows[-1]["id"], outcome, rows[-1]["latency_s"])
    summary = {
        "model": MODEL, "measured_at": datetime.now(timezone.utc).isoformat(), "target": target,
        "temperature": TEMPERATURE, "n": len(rows),
        "agree_with_label": sum(r["outcome"] == r["label"] for r in rows),
        "latency_median_s": st.median(r["latency_s"] for r in rows),
        "latency_max_s": max(r["latency_s"] for r in rows),
        "cost_per_1000_usd": 1000 * st.mean(r["cost_usd"] for r in rows),
        "rows": rows,
    }
    (DIR / "gemini-aligned.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print({k: v for k, v in summary.items() if k != "rows"})


main()
