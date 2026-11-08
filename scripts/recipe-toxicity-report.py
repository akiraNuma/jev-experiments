#!/usr/bin/env python3
"""実験3。保存した300件のうち、しきい値の調整に使っていない後半の文章で、有害投稿の見逃しと保留中の誤検出を比べる。

しきい値（recipe-toxicity-tune.py）はデータの並び順で前半を使って選んだ。300件は全体から一定間隔で選んだので、
約半分が前半と重なる。JevとDecisions APIだけが有利にならないよう、後半だけで比べる（F-047、2026-11-06）。
"""
import json
import math
import statistics
from pathlib import Path

DIR = Path("experiments/recipes/toxicity-ja")


def load(mode):
    path = DIR / f"results-{mode}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"duplicate IDs: {path}")
    return by_id


def toxic_probability(result):
    probabilities = result["label_probabilities"]
    if isinstance(probabilities, dict):
        return probabilities["toxic"]
    return next(row["probability"] for row in probabilities if row["value"] == "toxic")


baseline = load("llm-haiku")
if baseline is None:
    raise SystemExit("Haikuの保存結果がない")
data_ids = [str(json.loads(line)["id"]) for line in (DIR / "data/toxicity_dataset.jsonl").read_text().split("\n") if line.strip()]  # tune と同じく \n だけで分ける
tuning = set(data_ids[: len(data_ids) // 2])
ids = set(baseline) - tuning
print(f"比較: 保存した {len(baseline)} 件のうち、調整に使っていない後半 {len(ids)} 件")
holds = {}
for mode in ("jev", "decisions", "llm-haiku", "llm-sonnet"):
    rows = load(mode)
    if rows is None:
        continue
    if not ids <= set(rows):
        raise ValueError(f"{mode}: 比べる文章が揃っていない")
    # 調整したしきい値は、recipe-toxicity-tune.py で両方式とも同じ手順で選んだ値
    tuned = {"jev": 0.06, "decisions": 0.03}.get(mode)
    for method in (("choice", f"probability-{tuned}") if tuned is not None else ("choice",)):
        cases = [rows[key] for key in sorted(ids)]
        failures = sum(row["result"].get("ok") is False for row in cases)
        if failures:
            print(f"{mode} {method}: {failures} 件失敗。採点を保留")
            continue
        predicted = [row["result"]["label"] == "toxic" if method == "choice"
                     else toxic_probability(row["result"]) >= tuned for row in cases]
        actual = [row["gold"]["label"] == "toxic" for row in cases]
        holds[f"{mode} {method}"] = predicted
        toxic_count = sum(actual)
        held = sum(predicted)
        missed = sum(gold and not hold for gold, hold in zip(actual, predicted))
        false_held = sum(not gold and hold for gold, hold in zip(actual, predicted))
        print(f"{mode} {method}: 見逃し {missed}/{toxic_count} ({missed/toxic_count:.1%})、"
              f"保留中の誤検出 {false_held}/{held} ({false_held/held:.1%})、対象 {len(cases)} 件")
    costs = [row["result"].get("cost_usd") for row in rows.values() if row["id"] in ids]
    durations = [row["result"].get("duration_ms") for row in rows.values() if row["id"] in ids]
    if all(isinstance(value, (int, float)) for value in costs + durations):
        print(f"  1,000件あたり概算 ${statistics.mean(costs)*1000:.4f}、待ち中央値 {statistics.median(durations):.0f}ms")


def exact_p(b, c):
    n = b + c
    return min(1, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n) if n else 1.0


# 同じ文章への判定なので、1件ずつの対応のある比較（McNemarの正確検定）。
# 見逃しは有害な文章だけ、誤検出は有害でない文章を保留したかで比べる
cases = [baseline[key] for key in sorted(ids)]
actual = [row["gold"]["label"] == "toxic" for row in cases]
reference = "jev probability-0.06"
for other in [k for k in holds if k != reference and not k.startswith("jev ")]:
    for part, want in (("見逃し", True), ("有害でない文章の保留", False)):
        idx = [i for i, gold in enumerate(actual) if gold == want]
        # 見逃しは「保留しなかった」、誤検出は「保留した」が誤り
        err = (lambda h: not h) if want else (lambda h: h)
        b = sum(err(holds[reference][i]) and not err(holds[other][i]) for i in idx)
        c = sum(err(holds[other][i]) and not err(holds[reference][i]) for i in idx)
        print(f"検定 {reference} と {other}、{part}（{len(idx)} 件）: Jevだけ誤り {b}、相手だけ誤り {c}、p={exact_p(b, c):.4f}")
