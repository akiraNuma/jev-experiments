#!/usr/bin/env python3
"""実験3：保存済みの判定から、公開を保留する「有害」の確率のしきい値を選び直す。

    python3 scripts/recipe-toxicity-tune.py            # Jev
    python3 scripts/recipe-toxicity-tune.py decisions  # Decisions API（同じ前半で、同じ手順）

全3,847件の前半（データの順で先頭の半分）を調整に使い、0.01から0.99まで0.01刻みで試す（範囲の端で止まらない細かさ）。
「有害（toxic）」を正解とし、見逃しと誤検出の両方を反映するF1が最も高いしきい値を選ぶ。
APIは呼ばない。
"""
import json
import sys
from pathlib import Path

DIR = Path("experiments/recipes/toxicity-ja")
MODE = sys.argv[1] if len(sys.argv) > 1 else "jev"
GRID = [i / 100 for i in range(1, 100)]

data_ids = [str(json.loads(line)["id"]) for line in (DIR / "data/toxicity_dataset.jsonl").read_text().split("\n") if line.strip()]  # JS側と同じく \n だけで分ける
first_half = data_ids[: len(data_ids) // 2]
by_id = {}
for line in (DIR / f"results-{MODE}.jsonl").read_text().splitlines():
    if line.strip():
        row = json.loads(line)
        by_id[row["id"]] = row
missing = [id for id in first_half if id not in by_id or by_id[id]["result"].get("ok") is False]
if missing:
    raise SystemExit(f"{MODE}: 前半 {len(first_half)} 件のうち {len(missing)} 件の判定がない")
rows = list(by_id.values())
half = [by_id[id] for id in first_half]


def toxic_probability(row):
    probabilities = row["result"]["label_probabilities"]
    if isinstance(probabilities, dict):
        return probabilities["toxic"]
    return next(entry["probability"] for entry in probabilities if entry["value"] == "toxic")


def f1(threshold):
    held = [toxic_probability(r) >= threshold for r in half]
    gold = [r["gold"]["label"] == "toxic" for r in half]
    tp = sum(h and g for h, g in zip(held, gold))
    fp = sum(h and not g for h, g in zip(held, gold))
    fn = sum(g and not h for h, g in zip(held, gold))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


scores = {t: f1(t) for t in GRID}
best = max(GRID, key=lambda t: (scores[t], -t))
print(json.dumps({
    "mode": MODE,
    "total": len(rows),
    "tuning_n": len(half),
    "grid": {"min": GRID[0], "max": GRID[-1], "step": round(GRID[1] - GRID[0], 2)},
    "best": best,
    "f1": {f"{t:.2f}": round(scores[t], 4) for t in GRID},
}, ensure_ascii=False, indent=2))
