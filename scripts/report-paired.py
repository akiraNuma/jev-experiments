#!/usr/bin/env python3
"""【計測用】実験11の方式どうしの差を、保存結果から検定する。API通信はしない。

ゲームは方式ごとに別の回なので、対応のない比較（並べ替え検定、乱数の種は固定）。
"""
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOOM = ROOT / "experiments/recipes/doom"


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def permutation_p(left, right, trials=200_000):
    observed = abs(st.mean(left) - st.mean(right))
    pooled = left + right
    rng = random.Random(0)
    hits = 0
    for _ in range(trials):
        rng.shuffle(pooled)
        hits += abs(st.mean(pooled[:len(left)]) - st.mean(pooled[len(left):])) >= observed - 1e-12
    return hits / trials


def doom():
    for label, suffix in (("ゲームを進める", ""), ("ゲームを止めて待つ", "-sync")):
        rule = load(DOOM / f"results-rule{suffix}.jsonl")
        jev = load(DOOM / f"results-jev-v2{suffix}.jsonl")
        dec = load(DOOM / f"results-decisions-own-v2{suffix}.jsonl")
        kills = {name: [r["kills"] for r in rows] for name, rows in (("rule", rule), ("jev-v2", jev), ("decisions-own-v2", dec))}
        print(f"## 実験11 {label}: " + "、".join(f"{k} {st.mean(v):.1f}体（{len(v)}回）" for k, v in kills.items()))
        for a, b in (("jev-v2", "rule"), ("jev-v2", "decisions-own-v2"), ("decisions-own-v2", "rule")):
            print(f"  - {a} と {b}: p={permutation_p(kills[a], kills[b]):.4f}")
        print(f"  - decisions-own-v2: 1ゲームの平均費用 ${st.mean(r['decisions_cost_usd'] for r in dec):.4f}、"
              f"判定 平均 {st.mean(r['decisions'] for r in dec):.0f} 回、{st.mean(r['decisions_per_second'] for r in dec):.2f} 回/秒、"
              f"待ち 中央値の平均 {st.mean(r['latency_ms_median'] for r in dec):.0f}ms")


doom()
