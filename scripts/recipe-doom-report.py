#!/usr/bin/env python3
"""実験11。保存済みのゲーム結果を方式ごとに集計する。"""
import json
import statistics
from pathlib import Path

DIR = Path("experiments/recipes/doom")

for sync in (True, False):
    suffix = "-sync" if sync else ""
    print(f"{'同期' if sync else '実時間'}モード")
    for mode in ("rule", "jev-v2", "decisions", "decisions-v1", "decisions-own-v2", "llm-haiku", "random"):
        path = DIR / f"results-{mode}{suffix}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows:
            continue
        episodes = [row["episode"] for row in rows]
        if len(set(episodes)) != len(episodes):
            raise ValueError(f"duplicate episode IDs: {path}")
        rewards = statistics.mean(row["reward"] for row in rows)
        kills = statistics.mean(row["kills"] for row in rows)
        speed = statistics.mean(row["decisions_per_second"] for row in rows)
        refusals = sum(row.get("refusals", 0) for row in rows)
        print(f"  {mode}: {len(rows)} 回、平均報酬 {rewards:.1f}、平均撃破 {kills:.1f}、"
              f"判定 {speed:.2f} 回/秒、拒否 {refusals} 回")

# 著者が手で遊んだ記録（scripts/play-doom-human.py）。人は一時停止できないので実時間だけ
human = DIR / "results-human.jsonl"
if human.exists():
    rows = [json.loads(line) for line in human.read_text().splitlines() if line]
    kills = [row["kills"] for row in rows]
    print("人が遊んだ記録（実時間）")
    print(f"  human: {len(rows)} 回、平均撃破 {statistics.mean(kills):.1f}、"
          f"倒された回 {sum(row['died'] for row in rows)}、各回の撃破 {','.join(map(str, kills))}")
