"""【計測用】実験8。JevまたはDecisionsの確信度でClaudeへ回す件を決める。

保存結果の共通対象へ揃えて組み合わせる。新しいAPI通信は行わない。
回答拒否は除かず不正解として数える。先に使う方式の拒否は確信度がないのでHaikuへ回す（findings F-047）。
"""
import json
from pathlib import Path

DIR = Path("experiments/classify")


def load(dataset, mode):
    path = DIR / f"results-{dataset}-{mode}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"duplicate IDs: {path}")
    return by_id


# classify-judge.ts は Decisions API の refusal にこの誤りを付ける
REFUSAL_ERRORS = {"unexpected intent answer"}


def refused(row):
    result = row["result"]
    return result.get("ok") is False and (result.get("error") in REFUSAL_ERRORS or bool(result.get("refused_questions")))


def correct(row):
    return row["result"].get("ok", True) and row["result"].get("predicted") == row["label"]


for dataset in ("banking77", "massive-ja"):
    groups = {mode: load(dataset, mode) for mode in ("jev", "decisions", "llm-haiku", "llm-sonnet", "tfidf")}
    if any(rows is None for rows in groups.values()):
        raise ValueError(f"掲載条件の5方式の保存結果が必要です: {dataset}")
    haiku = groups["llm-haiku"]
    sonnet = groups["llm-sonnet"]
    ids = set(haiku)
    ids = {id for id in ids if all(id in rows and (rows[id]["result"].get("ok") is not False or refused(rows[id]))
                                   for rows in groups.values())}
    if not ids:
        raise ValueError(f"共通の評価対象がありません: {dataset}")
    for id in ids:
        if any(rows[id]["label"] != haiku[id]["label"] for rows in groups.values()):
            raise ValueError(f"正解ラベルが異なります: {dataset} {id}")
    for first_mode in ("jev", "decisions"):
        first = groups[first_mode]
        ordered = sorted(ids)
        n = len(ordered)
        print(f"{dataset} {first_mode}: {n} 件、単独正解 {sum(correct(first[i]) for i in ordered)}/{n}、Haiku単独 {sum(correct(haiku[i]) for i in ordered)}/{n}")
        for threshold in (0.5, 0.7, 0.8, 0.9, 0.95):
            escalated = [i for i in ordered if first[i]["result"].get("ok") is False
                         or first[i]["result"].get("confidence", 0) < threshold]
            escalated_ids = set(escalated)
            haiku_right = sum(correct(haiku[i] if i in escalated_ids else first[i]) for i in ordered)
            # 古い拒否の記録には費用がないので、費用は成功した行だけで平均する
            first_costs = [first[i]["result"].get("cost_usd") for i in ordered if not refused(first[i])]
            fallback_costs = [haiku[i]["result"].get("cost_usd") for i in escalated]
            cost = (sum(first_costs) * 1000 / len(first_costs) + sum(fallback_costs) * 1000 / n) if first_costs and all(
                isinstance(v, (int, float)) for v in first_costs + fallback_costs) else None
            line = f"  線 {threshold}: Haikuへ {len(escalated)}/{n}、正解 {haiku_right}/{n}"
            if cost is not None:
                line += f"、1,000件あたり ${cost:.4f}（保存結果の費用・Claude Codeの共通入力込み）"
            if sonnet is not None:
                sonnet_right = sum(correct(sonnet[i] if i in escalated_ids else first[i]) for i in ordered)
                line += f"、Sonnetへ回した場合 {sonnet_right}/{n}"
            print(line)
