#!/usr/bin/env python3
"""保存結果から、同じ評価対象に揃えた書籍の比較表を再集計する。API通信はしない。

回答拒否は、どの方式でも不正解として数え、集計から除かない（2026-11-06 著者の決定、findings F-047）。
- 1件1ラベルの分類: 不正解。確信度がないので「確信度0.9以上だけ採用」では確認待ち、カスケードではHaikuへ回す
- 複数ラベル: 拒否した感情は「付けなかった」として数える（正解なら見逃し）。回答が保存されていない行は、
  どの感情も付けなかったとみなす（アプリではそのコメントにラベルが付かないため）
"""
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}


def load(name, key="id"):
    path = ROOT / "experiments" / name
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate IDs: {name}")
    return result


# 測定スクリプトが回答拒否（Decisions APIの refusal）に付けた誤りの印。decisions-api.ts は refusal 以外の
# 想定外の型を例外にするので、この印の行は拒否。新しい記録では refused_questions に拒否した質問が残る。
REFUSAL_ERRORS = {"unexpected intent answer", "unexpected predicate answer"}


def ok(row):
    return row.get("result", row).get("ok") is not False


def refused(row):
    result = row.get("result", row)
    return result.get("ok") is False and (result.get("error") in REFUSAL_ERRORS or bool(result.get("refused_questions")))


def common(groups, reference):
    """全方式で、成功か回答拒否のどちらかが記録された ID。拒否は不正解として残し、通信失敗などは除く。"""
    return sorted(id for id in reference if all(id in rows and (ok(rows[id]) or refused(rows[id]))
                                                for rows in groups.values()))


def toxicity_data_ids():
    """有害判定データのIDを、データの並び順で返す。recipe-toxicity-tune.py と同じく \\n だけで分ける。"""
    path = ROOT / "experiments/recipes/toxicity-ja/data/toxicity_dataset.jsonl"
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return [str(json.loads(line)["id"]) for line in data.decode().split("\n") if line.strip()]


def toxicity_heldout(ids, data_ids):
    """しきい値の調整に使っていない後半の文章だけを残す。

    しきい値は前半（データの並び順で先頭の半分）で選んだ。比較の300件は全体から一定間隔で選んだので、
    約半分が前半と重なり、JevとDecisions APIだけが有利になる。比較は後半だけで行う（F-047、2026-11-06）。
    """
    tuning = set(data_ids[: len(data_ids) // 2])
    return [id for id in ids if id not in tuning]


def correct(row):
    return ok(row) and row["result"]["predicted"] == row["label"]


def confidence(row):
    """回答拒否には確信度がないので None。しきい値の比較では常に下回る側に入れる。"""
    return row["result"]["confidence"] if ok(row) else None


def sure(row, threshold=.9):
    return confidence(row) is not None and confidence(row) >= threshold


def usage(rows):
    # 費用・待ち時間は、利用量が記録された成功行で計算する（古い拒否の記録には費用がない）
    selected = [r.get("result", r) for r in rows if ok(r) and not r.get("result", r).get("skipped")]
    return {"calls": len(selected), "cost_per_1000": st.mean(r["cost_usd"] for r in selected)*1000,
            "latency_ms_median": st.median(r.get("duration_ms", r.get("wall_ms")) for r in selected)}


def exact_p(left, right):
    b = sum(a and not c for a, c in zip(left, right))
    c = sum(d and not a for a, d in zip(left, right))
    n = b + c
    return min(1, 2*sum(math.comb(n, k) for k in range(min(b, c)+1))/2**n) if n else 1


def emotion_prediction(row, emotions, threshold=None):
    """付けた感情の集合。拒否した感情は付けなかったとして数える（正解にあれば見逃し、誤検出にはしない）。

    回答が保存されていない拒否の行は、どの感情も付けなかったとみなす。
    1つも付かなければ、拒否していない感情のうち確率が最も高いものを1つ付ける（掲載コードと同じ）。
    """
    result = row["result"]
    if ok(row):
        if threshold is None:
            return set(result["predicted"])
        scores = result["scores"]
        return {e for e, v in scores.items() if v >= threshold} or {max(scores, key=scores.get)}
    if not refused(row):
        raise ValueError(f"unexpected failure: {row['id']}")
    answers = result.get("answers") or {}
    scores = {e: a["probability"] for e, a in answers.items() if a.get("type") == "predicate"}
    # 確率のない方式（Claude）や回答が保存されていない行では scores が空なので、何も付かない
    pred = {e for e, v in scores.items() if v >= (threshold or .5)}
    return pred or ({max(scores, key=scores.get)} if scores else set())


def emotion_metrics(rows, threshold=None, emotions=None):
    if emotions is None:
        emotions = sorted({e for r in rows if ok(r) and r["result"].get("scores") for e in r["result"]["scores"]})
    predictions = [emotion_prediction(r, emotions, threshold) for r in rows]
    exact = [pred == set(row["labels"]) for row, pred in zip(rows, predictions)]
    tp = sum(len(pred & set(row["labels"])) for row, pred in zip(rows, predictions))
    fp = sum(len(pred - set(row["labels"])) for row, pred in zip(rows, predictions))
    fn = sum(len(set(row["labels"]) - pred) for row, pred in zip(rows, predictions))
    # 感情ごとのF1。正解と予測のどちらにも現れない感情は載せない。
    per_label = {}
    for label in sorted(set().union(*predictions, *(set(r["labels"]) for r in rows))):
        ltp = sum(label in pred and label in row["labels"] for row, pred in zip(rows, predictions))
        lfp = sum(label in pred and label not in row["labels"] for row, pred in zip(rows, predictions))
        lfn = sum(label not in pred and label in row["labels"] for row, pred in zip(rows, predictions))
        per_label[label] = {"gold": ltp+lfn, "predicted": ltp+lfp, "f1": 2*ltp/(2*ltp+lfp+lfn)}
    counts = [(len(pred & set(row["labels"])), len(pred - set(row["labels"])), len(set(row["labels"]) - pred))
              for row, pred in zip(rows, predictions)]
    return {"n": len(rows), "exact": sum(exact), "accuracy": st.mean(exact),
            "micro_f1": 2*tp/(2*tp+fp+fn), "correct_by_id_order": exact, "per_label": per_label,
            "_counts": counts}


def micro_f1_p(left, right, trials=20_000):
    """コメントごとの (tp, fp, fn) を2方式で入れ替える並べ替え検定。micro F1 の差の両側 p。乱数の種は固定。"""
    import random

    def f1_of(counts):
        tp, fp, fn = (sum(c[i] for c in counts) for i in range(3))
        return 2*tp/(2*tp+fp+fn)
    observed = abs(f1_of(left) - f1_of(right))
    rng = random.Random(0)
    hits = 0
    for _ in range(trials):
        a, b = [], []
        for x, y in zip(left, right):
            if rng.random() < .5:
                x, y = y, x
            a.append(x)
            b.append(y)
        hits += abs(f1_of(a) - f1_of(b)) >= observed - 1e-12
    return hits / trials


def report():
    out = {"classify": {}, "cascade": {}, "emotions": {}, "toxicity": {}, "sources": SOURCES}
    for dataset in ("banking77", "massive-ja"):
        groups = {mode: load(f"classify/results-{dataset}-{mode}.jsonl")
                  for mode in ("jev", "decisions", "llm-haiku", "llm-sonnet", "tfidf")}
        ids = common(groups, groups["llm-haiku"])
        table = {"ids": ids, "n": len(ids), "modes": {}}
        for mode, data in groups.items():
            rows = [data[id] for id in ids]
            if any(r["label"] != groups["jev"][r["id"]]["label"] for r in rows):
                raise ValueError("gold labels differ")
            summary = {"correct": sum(correct(r) for r in rows), "refused": sum(refused(r) for r in rows)}
            if mode in ("jev", "decisions"):
                accepted = [r for r in rows if sure(r)]
                summary.update(usage(rows), accepted=len(accepted), held=len(rows)-len(accepted),
                               accepted_correct=sum(correct(r) for r in accepted))
            table["modes"][mode] = summary
        # 片方だけが正解した件数とMcNemar正確検定のp値（Jevと各方式）
        right = {mode: [correct(data[id]) for id in ids] for mode, data in groups.items()}
        table["pairs"] = {mode: {"jev_only": sum(a and not b for a, b in zip(right["jev"], right[mode])),
                                 "other_only": sum(b and not a for a, b in zip(right["jev"], right[mode])),
                                 "p": exact_p(right["jev"], right[mode])}
                          for mode in groups if mode != "jev"}
        out["classify"][dataset] = table
        out["cascade"][dataset] = {"n": len(ids), "modes": {}}
        for mode in ("jev", "decisions"):
            first = groups[mode]
            # 回答拒否は確信度がないのでHaikuへ回す
            sent = [id for id in ids if not sure(first[id])]
            routed = [correct(groups["llm-haiku"][id] if id in set(sent) else first[id]) for id in ids]
            alone = [correct(groups["llm-haiku"][id]) for id in ids]
            # 全件をHaikuで分類した場合との差（McNemar）
            out["cascade"][dataset]["modes"][mode] = {"sent": len(sent), "correct": sum(routed),
                                                     "vs_haiku_p": exact_p(routed, alone)}
        out["cascade"][dataset]["haiku_correct"] = table["modes"]["llm-haiku"]["correct"]
    # 費用試算：Haikuの一覧をキャッシュし、本文0〜100トークン。単価は scripts/classify-cost.py の banking77 の値
    # （2026-11-02、個人設定を読ませない条件で測り直した tokens-haiku）。実測したAPI利用量からの費用とは分ける。
    sample = out["classify"]["banking77"]
    estimates = {}
    for mode in ("jev", "decisions"):
        first = sample["modes"][mode]["cost_per_1000"]
        ratio = out["cascade"]["banking77"]["modes"][mode]["sent"] / sample["n"]
        estimates[mode] = {"with_thinking": [first+ratio*x for x in (.092, .101)],
                           "without_thinking": [first+ratio*x for x in (.043, .052)]}
    out["cascade_cost_estimates"] = estimates
    groups = {m: load(f"recipes/goemotions/results-{m}.jsonl") for m in ("jev", "decisions", "llm-haiku", "llm-sonnet")}
    ids = common(groups, groups["llm-haiku"])
    modes = {}
    emotions = list(next(r for r in groups["jev"].values() if ok(r))["result"]["scores"])
    # 調整したしきい値は、両方式とも同じ dev・同じ手順で選んだ値（scripts/recipe-report.py の goemotions_tuned。
    # dev の回答拒否も、評価と同じく拒否した感情を付けなかったとして数えて選んだ）
    for name, mode, threshold in (("jev-0.5", "jev", .5), ("jev-0.92", "jev", .92),
                                  ("decisions-0.5", "decisions", .5), ("decisions-0.97", "decisions", .97),
                                  ("llm-haiku", "llm-haiku", None),
                                  ("llm-sonnet", "llm-sonnet", None)):
        rows = [groups[mode][id] for id in ids]
        if any(set(r["labels"]) != set(groups["jev"][r["id"]]["labels"]) for r in rows):
            raise ValueError("emotion gold differs")
        modes[name] = emotion_metrics(rows, threshold, emotions)
        modes[name]["refused"] = sum(refused(r) for r in rows)
        if mode in ("jev", "decisions"):
            modes[name].update(usage(rows))
    out["emotions"] = {"n": len(ids), "ids": ids, "modes": modes,
        "jev_tuned_vs_haiku_p": exact_p(modes["jev-0.92"]["correct_by_id_order"], modes["llm-haiku"]["correct_by_id_order"]),
        "jev_tuned_vs_sonnet_p": exact_p(modes["jev-0.92"]["correct_by_id_order"], modes["llm-sonnet"]["correct_by_id_order"]),
        # それぞれが dev で選んだしきい値どうしの比較
        "jev_tuned_vs_decisions_tuned_p": exact_p(modes["jev-0.92"]["correct_by_id_order"], modes["decisions-0.97"]["correct_by_id_order"]),
        # micro F1 はコメントごとの正誤ではないので、コメント単位で2方式の判定を入れ替えて検定する
        "micro_f1_p": {other: micro_f1_p(modes["jev-0.92"]["_counts"], modes[other]["_counts"])
                       for other in ("decisions-0.97", "llm-haiku", "llm-sonnet")}}
    for metrics in modes.values():
        metrics.pop("_counts")
    groups = {m: load(f"recipes/toxicity-ja/results-{m}.jsonl") for m in ("jev", "decisions", "llm-haiku", "llm-sonnet")}
    ids = toxicity_heldout(common(groups, groups["llm-haiku"]), toxicity_data_ids())
    # 見逃し・誤検出は各方式の行の正解で数えるため、正解が方式間で同じかを先に確かめる。
    if any(groups[m][id]["gold"]["label"] != groups["jev"][id]["gold"]["label"] for m in groups for id in ids):
        raise ValueError("toxicity gold differs")
    toxic = sum(groups["jev"][id]["gold"]["label"] == "toxic" for id in ids)
    out["toxicity"] = {"n": len(ids), "toxic_n": toxic, "modes": {}}
    for mode in groups:
        # 「有害」の確率のしきい値は、データの前半で両方式とも同じ手順で選んだ値（scripts/recipe-toxicity-tune.py）
        tuned_threshold = {"jev": .06, "decisions": .03}.get(mode)
        for threshold in ([None, tuned_threshold] if tuned_threshold is not None else [None]):
            held = []
            for id in ids:
                if refused(groups[mode][id]):  # 拒否は正解と逆の判定（有害なら保留しない、有害でなければ保留する）
                    if groups[mode][id]["gold"]["label"] != "toxic":
                        held.append(id)
                    continue
                r = groups[mode][id]["result"]
                probs = r.get("label_probabilities")
                p = (probs["toxic"] if isinstance(probs, dict) else next(x["probability"] for x in probs if x["value"] == "toxic")) if threshold is not None else None
                if (r["label"] == "toxic" if threshold is None else p >= threshold):held.append(id)
            missed = sum(id not in held and groups[mode][id]["gold"]["label"] == "toxic" for id in ids)
            false = sum(groups[mode][id]["gold"]["label"] != "toxic" for id in held)
            out["toxicity"]["modes"][f"{mode}-{'choice' if threshold is None else 'tuned'}"] = {
                "threshold": threshold, "held": len(held), "missed": missed, "miss_rate": missed/toxic, "false_held": false,
                "false_held_rate": false/len(held)}
        if mode in ("jev", "decisions"):
            out["toxicity"]["modes"][f"{mode}-tuned"].update(usage([groups[mode][id] for id in ids]))
    return out


if __name__ == "__main__":
    print(json.dumps(report(), ensure_ascii=False, indent=2))
