#!/usr/bin/env python3
"""保存した結果だけを再集計する。API は呼ばない。"""
import json
import math
import statistics
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "experiments/recipes/pets"
CENTER = {"left": 0.25, "top": 0.25, "right": 0.75, "bottom": 0.75}


def iou(a, b):
    intersection = max(0, min(a["right"], b["right"]) - max(a["left"], b["left"])) * max(
        0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]))
    area = lambda box: max(0, box["right"] - box["left"]) * max(0, box["bottom"] - box["top"])
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    divisor = 1 + z * z / n
    center = (p + z * z / (2 * n)) / divisor
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / divisor
    return [center - half, center + half]


def mcnemar(a, b):
    a_only = sum(a[key] and not b[key] for key in a)
    b_only = sum(b[key] and not a[key] for key in a)
    n = a_only + b_only
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2 ** n) if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "p_exact_two_sided": p}


def summarize(mode, items):
    path = DIR / f"results-{mode}-eval.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != {item["id"] for item in items}:
        raise ValueError(f"duplicate/missing/unexpected evaluation IDs: {path}")
    results = [by_id[item["id"]]["result"] for item in items]
    good = [r for r in results if r["ok"]]
    n = len(items)
    summary = {"n": n, "success": len(good), "failures": n - len(good),
               "cost_usd_total_success": sum(r["cost_usd"] for r in good),
               "cost_usd_per_1000_success": statistics.mean(r["cost_usd"] for r in good) * 1000,
               "input_tokens_median": statistics.median(
                   r["input_tokens"] if mode.startswith("decisions-") else r["usage"]["input_tokens"] for r in good),
               "latency_ms_median": statistics.median(r["duration_ms"] for r in good),
               "latency_ms_max": max(r["duration_ms"] for r in good),
               "actual_models": sorted({r["model"] for r in good})}
    if mode.startswith("box"):
        scores = [iou(r["answers"]["head"]["box"], item["box_784"]) if r["ok"] else 0.0
                  for item, r in zip(items, results)]
        baseline = [iou(CENTER, item["box_784"]) for item in items]
        summary.update({"iou_median": statistics.median(scores), "iou_mean": statistics.mean(scores),
                        "iou_ge_0.5": sum(v >= 0.5 for v in scores),
                        "center_baseline_iou_median": statistics.median(baseline),
                        "center_baseline_iou_ge_0.5": sum(v >= 0.5 for v in baseline),
                        "pet_present_ge_0.5": sum(r["answers"]["pet_present"]["noul"] >= 0.5 for r in good),
                        "bicycle_present_ge_0.5": sum(r["answers"]["absent_present"]["noul"] >= 0.5 for r in good),
                        "bicycle_box_returned": sum("box" in r["answers"]["absent"] for r in good),
                        "iou_by_id": dict(zip([i["id"] for i in items], scores)),
                        "center_baseline": CENTER})
        return summary, None
    correct = {item["id"]: bool(r["ok"] and r["answers"]["breed"]["choice"] == item["breed"])
               for item, r in zip(items, results)}
    k = sum(correct.values())
    summary.update({"correct": k, "accuracy": k / n, "wilson_95": wilson(k, n),
                    "correct_by_breed": {breed: sum(correct[i["id"]] for i in items if i["breed"] == breed)
                                         for breed in sorted({i["breed"] for i in items})},
                    "wrong_ids": [key for key, value in correct.items() if not value]})
    return summary, correct


def main():
    items = [i for i in json.loads((DIR / "manifest.json").read_text())["items"] if i["split"] == "eval"]
    report = {"classification_baseline_always_one_breed": 1 / 6, "evaluation_n": len(items), "modes": {}}
    correctness = {}
    for mode in ("text-1.13", "text-1.14", "examples-1.13", "examples-1.14", "box-1.14"):
        summary, correct = summarize(mode, items)
        report["modes"][mode] = summary
        if correct is not None:
            correctness[mode] = correct
    for mode in ("decisions-text", "decisions-examples"):
        if (DIR / f"results-{mode}-eval.jsonl").exists():
            summary, correct = summarize(mode, items)
            report["modes"][mode] = summary
            correctness[mode] = correct
    report["paired"] = {
        "text1.13_vs_text1.14": mcnemar(correctness["text-1.13"], correctness["text-1.14"]),
        "text1.14_vs_examples1.14": mcnemar(correctness["text-1.14"], correctness["examples-1.14"]),
        "text1.13_vs_examples1.13": mcnemar(correctness["text-1.13"], correctness["examples-1.13"]),
    }
    for mode in ("decisions-text", "decisions-examples"):
        if mode in correctness:
            report["paired"][f"text1.14_vs_{mode}"] = mcnemar(correctness["text-1.14"], correctness[mode])
    (DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "modes": {k: {key: value for key, value in v.items()
                      if key != "iou_by_id"} for k, v in report["modes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
