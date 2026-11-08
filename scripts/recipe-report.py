"""【計測用】実験2と6の結果を集計する。

    python3 scripts/recipe-report.py

- LLM の費用は、claude -p の前置き（overhead-<model>.json）を差し引いた入力トークンと、公式の単価から計算する
- claude -p では考える段を切れないので、費用は「考える段込み」と「なし」の 2 通りを出す。正解率は考える段込みの値しか無い
- 差が偶然かは、同じ件で片方だけが正解した数を McNemar の正確検定に掛ける
- 回答拒否は集計から除かず、拒否した感情を付けなかったとして数える（調整用の dev も同じ。findings F-047）。
  回答が保存されていない拒否の行は、どの感情も付けなかったとみなす
"""

import json
import math
import statistics as st
from pathlib import Path

PRICES = {"haiku": (0.1, 0.5), "sonnet": (2, 10)}  # 100 万トークンあたり、入力と出力。2026-10-13 に料金表で確認
TOOL_PROMPT_TOKENS = 406


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


def llm_costs(rows, overhead, model):
    pi, po = PRICES[model]
    without, with_think = [], []
    for r in rows:
        x = r["result"]
        prompt = x["input_total"] - overhead + TOOL_PROMPT_TOKENS
        out = x["output_tokens"] - x["thinking_tokens"]
        without.append((prompt * pi + out * po) / 1e6)
        with_think.append((prompt * pi + x["output_tokens"] * po) / 1e6)
    return 1000 * st.mean(without), 1000 * st.mean(with_think), st.median(x["result"]["thinking_tokens"] for x in rows)


def sign_flip_p(diffs, trials=200_000):
    """対応のある差の、符号反転による両側検定。乱数の種は固定。差0は除く。"""
    import random
    diffs = [d for d in diffs if d]
    observed = abs(sum(diffs))
    rng = random.Random(0)
    hits = sum(abs(sum(d if rng.random() < .5 else -d for d in diffs)) >= observed for _ in range(trials))
    return len(diffs), hits / trials


# 測定スクリプトが回答拒否に付けた誤りの印（scripts/report-decisions-book.py と同じ判定）
REFUSAL_ERRORS = {"unexpected intent answer", "unexpected predicate answer"}


def refused(r):
    x = r["result"]
    return x.get("ok") is False and (x.get("error") in REFUSAL_ERRORS or bool(x.get("refused_questions")))


def emotion_scores(r):
    """拒否していない感情の確率。回答が保存されていない拒否の行は空。"""
    x = r["result"]
    if x.get("ok") is not False:
        return x["scores"]
    return {e: a["probability"] for e, a in (x.get("answers") or {}).items() if a.get("type") == "predicate"}


def emotion_pred(r, emotions, thresholds=None):
    """付けた感情の集合。thresholds が None なら保存した predicted（線 0.5、補いなし）を使う。

    拒否した感情は付けなかったとして数える（正解にあれば見逃し、誤検出にはしない）。
    線を使う場合、1つも付かなければ拒否していない感情のうち確率が最も高いものを1つ付ける。
    """
    x = r["result"]
    if x.get("ok") is not False:
        if thresholds is None:
            return set(x["predicted"])
        return set(tuned(x["scores"], thresholds))
    if not refused(r):
        raise ValueError(f"回答拒否以外の失敗: {r['id']}")
    scores = emotion_scores(r)
    line = thresholds or {e: 0.5 for e in emotions}
    pred = {e for e, v in scores.items() if v >= line[e]}
    if thresholds is not None and not pred and scores:
        pred = {max(scores, key=scores.get)}
    return pred


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def goemotions():
    d = Path("experiments/recipes/goemotions")
    print("## GoEmotions（28 種類、複数可）")
    jev = {r["id"]: r for r in load(d / "results-jev.jsonl") if r["result"]["ok"]}
    emotions = list(next(iter(jev.values()))["result"]["scores"])
    modes = ["jev", "llm-haiku", "llm-sonnet"]
    if (d / "results-decisions.jsonl").exists():
        modes.insert(1, "decisions")
    for mode in modes:
        recorded = load(d / f"results-{mode}.jsonl")
        rows = [r for r in recorded if r["result"]["ok"] or refused(r)]
        failures = len(recorded) - len(rows)
        if failures:
            print(f"- {mode}: 対象 {len(recorded)} 件、回答拒否以外の失敗 {failures} 件。全対象の比較採点を保留")
            continue
        n_refused = sum(refused(r) for r in rows)
        if n_refused:
            print(f"- {mode}: 回答拒否 {n_refused} 件は、拒否した感情を付けなかったとして数える")
        ids = {r["id"] for r in load(d / "results-llm-haiku.jsonl")}
        sample = [r for r in rows if r["id"] in ids]
        first_label = "測定済み" if mode == "decisions" else "全件"
        for name, rs in [(first_label, rows), ("LLM と同じ 300 件", sample)]:
            if mode.startswith("llm-") and name == "全件":
                continue
            tp = fp = fn = 0
            per = {e: [0, 0, 0] for e in emotions}
            exact = 0
            for r in rs:
                gold, pred = set(r["labels"]), emotion_pred(r, emotions)
                exact += gold == pred
                for e in emotions:
                    k = 0 if (e in gold and e in pred) else 1 if (e in pred) else 2 if (e in gold) else None
                    if k is not None:
                        per[e][k] += 1
            tp, fp, fn = (sum(v[i] for v in per.values()) for i in range(3))
            macro = st.mean(f1(*v) for v in per.values())
            print(f"- {mode} {name}: micro F1 {f1(tp, fp, fn):.3f}、macro F1 {macro:.3f}、完全一致 {exact / len(rs):.1%}（{len(rs)} 件）")
        if mode in ("jev", "decisions"):
            # 費用・待ちは利用量が記録された成功行で計算する（古い拒否の記録には費用がない）
            used = [r for r in rows if r["result"]["ok"]]
            print(f"  - 1,000 件あたり ${1000 * st.mean(r['result']['cost_usd'] for r in used):.3f}、待ち 中央値 {st.median(r['result']['duration_ms'] for r in used)}ms（成功 {len(used)} 件）")
            if mode == "decisions":
                both = [r for r in rows if r["id"] in jev]
                ok = lambda r: set(r["labels"]) == emotion_pred(r, emotions)  # noqa: E731
                bj = sum(1 for r in both if ok(jev[r["id"]]) and not ok(r))
                bd = sum(1 for r in both if ok(r) and not ok(jev[r["id"]]))
                print(f"  - 完全一致で Jev だけ正解 {bj} / Decisions だけ正解 {bd}、p={mcnemar(bj, bd):.3f}")
        else:
            model = mode.split("-")[1]
            overhead = json.load(open(d / f"overhead-{model}.json"))["tokens"]
            a, b, think = llm_costs(rows, overhead, model)
            print(f"  - 1,000 件あたり（API 換算）考える段なし ${a:.3f}、考える段込み ${b:.3f}（考える段 中央値 {think} トークン）")
            # 1 件の完全一致で、Jev と比べる
            both = [r for r in rows if r["id"] in jev]
            ok = lambda r: set(r["labels"]) == set(r["result"]["predicted"])  # noqa: E731
            bj = sum(1 for r in both if ok(jev[r["id"]]) and not ok(r))
            bl = sum(1 for r in both if ok(r) and not ok(jev[r["id"]]))
            print(f"  - 完全一致で Jev だけ正解 {bj} / LLM だけ正解 {bl}、p={mcnemar(bj, bl):.3f}")


def tuned(scores, thresholds):
    """線以上の感情を返す。1 つも無ければ一番高いものを 1 つ返す（正解は必ず 1 つ以上ある）"""
    picked = [e for e, v in scores.items() if v >= thresholds[e]]
    return picked or [max(scores, key=scores.get)]


def goemotions_tuned(mode="jev"):
    """dev で線を決めて、test に当てる。線は 0.01〜0.99 を 0.01 刻みで試し、dev の F1 が一番高いもの。
    範囲の端で止まると範囲外によい値が残るため、端に当たらない細かさにしている。

    Decisions API にも同じ dev・同じ手順で線を選ぶ（mode="decisions"）。
    """
    d = Path("experiments/recipes/goemotions")
    if not (d / f"results-{mode}-dev.jsonl").exists():
        return
    dev_all = load(d / f"results-{mode}-dev.jsonl")
    # 回答拒否のあったコメントも除かず、拒否した感情を付けなかったとして数える。それ以外の失敗が残っていれば線を選ばない
    dev = [r for r in dev_all if r["result"]["ok"] or refused(r)]
    n_refused = sum(refused(r) for r in dev)
    if len(dev) != len(dev_all):
        print(f"\n## GoEmotions、{mode}: dev の回答拒否以外の失敗 {len(dev_all) - len(dev)} 件。線の選択を保留")
        return
    emotions = list(next(r for r in dev if r["result"]["ok"])["result"]["scores"])
    test_all = load(d / f"results-{mode}.jsonl")
    test = [r for r in test_all if r["result"]["ok"] or refused(r)]
    if len(test) != len(test_all):
        print(f"\n## GoEmotions、{mode}: 評価の回答拒否以外の失敗 {len(test_all) - len(test)} 件。採点を保留")
        return
    grid = [i / 100 for i in range(1, 100)]

    def score(rows, thresholds):
        per = {e: [0, 0, 0] for e in emotions}
        exact = 0
        for r in rows:
            gold, pred = set(r["labels"]), emotion_pred(r, emotions, thresholds)
            exact += gold == pred
            for e in emotions:
                if e in gold and e in pred:
                    per[e][0] += 1
                elif e in pred:
                    per[e][1] += 1
                elif e in gold:
                    per[e][2] += 1
        tp, fp, fn = (sum(v[i] for v in per.values()) for i in range(3))
        return f1(tp, fp, fn), st.mean(f1(*v) for v in per.values()), exact / len(rows)

    best_global = max(grid, key=lambda t: score(dev, {e: t for e in emotions})[0])
    per_emotion = {}
    for e in emotions:
        def f1_e(t):
            # 拒否した感情は、線によらず付けなかったとして数える
            picked = [(e in r["labels"], e in emotion_scores(r) and emotion_scores(r)[e] >= t)
                      for r in dev]
            tp = sum(1 for gold, pred in picked if gold and pred)
            fp = sum(1 for gold, pred in picked if not gold and pred)
            fn = sum(1 for gold, pred in picked if gold and not pred)
            return f1(tp, fp, fn)
        per_emotion[e] = max(grid, key=f1_e)
    print(f"\n## GoEmotions、線を dev で決めた {mode}（dev {len(dev)} 件、うち回答拒否を付けなかったとして数えた {n_refused} 件、全体の線 {best_global}）")
    ids = {r["id"] for r in load(d / "results-llm-haiku.jsonl")}
    llm = {m: {r["id"]: r for r in load(d / f"results-{m}.jsonl") if r["result"]["ok"] or refused(r)} for m in ["llm-haiku", "llm-sonnet"]}
    for name, th in [("全体で 1 本の線", {e: best_global for e in emotions}), ("感情ごとの線", per_emotion)]:
        for label, rows in [("全件", test), ("LLM と同じ 300 件", [r for r in test if r["id"] in ids])]:
            mi, ma, ex = score(rows, th)
            print(f"- {name}、{label}: micro F1 {mi:.3f}、macro F1 {ma:.3f}、完全一致 {ex:.1%}")
        test_ids = {t["id"] for t in test}
        for m, rows in llm.items():
            rows = {i: r for i, r in rows.items() if i in test_ids}
            by_id = {t["id"]: t for t in test}
            mine = {i: set(r["labels"]) == emotion_pred(by_id[i], emotions, th) for i, r in rows.items()}
            other = {i: set(r["labels"]) == emotion_pred(r, emotions) for i, r in rows.items()}
            bj = sum(1 for i in rows if mine[i] and not other[i])
            bl = sum(1 for i in rows if other[i] and not mine[i])
            print(f"  - 完全一致で {mode} だけ正解 {bj} / {m} だけ正解 {bl}、p={mcnemar(bj, bl):.3f}")


def contract():
    d = Path("experiments/recipes/contractnli")
    print("\n## ContractNLI（契約書 123 通 × 17 条項、3 択）")
    jev = {r["id"]: r for r in load(d / "results-jev.jsonl") if r["result"]["ok"]}
    modes = ["jev", "jev-separate", "llm-haiku", "llm-sonnet"]
    if (d / "results-decisions.jsonl").exists():
        modes.insert(2, "decisions")
    for mode in modes:
        recorded = load(d / f"results-{mode}.jsonl")
        rows = [r for r in recorded if r["result"]["ok"]]
        failures = len(recorded) - len(rows)
        if failures:
            print(f"- {mode}: 対象 {len(recorded)} 件、成功 {len(rows)} 件、失敗 {failures} 件。全対象の比較採点を保留")
            continue
        n = c = 0
        for r in rows:
            for k, g in r["gold"].items():
                n += 1
                c += r["result"]["predicted"].get(k) == g
        print(f"- {mode}: 正解率 {c / n:.1%}（{c}/{n}、{len(rows)} 通）")
        if mode.startswith("jev") or mode == "decisions":
            print(f"  - 1,000 通あたり ${1000 * st.mean(r['result']['cost_usd'] for r in rows):.3f}、入力 中央値 {st.median(r['result']['input_tokens'] for r in rows)} トークン、待ち 中央値 {st.median(r['result']['duration_ms'] for r in rows)}ms")
            if mode == "decisions":
                bj = bd = 0
                for r in rows:
                    for k, g in r["gold"].items():
                        j = jev[r["id"]]["result"]["predicted"].get(k) == g
                        other = r["result"]["predicted"].get(k) == g
                        bj += j and not other
                        bd += other and not j
                print(f"  - 条項ごとに Jev だけ正解 {bj} / Decisions だけ正解 {bd}、p={mcnemar(bj, bd):.4f}")
        else:
            model = mode.split("-")[1]
            overhead = json.load(open(d / f"overhead-{model}.json"))["tokens"]
            a, b, think = llm_costs(rows, overhead, model)
            print(f"  - 1,000 通あたり（API 換算）考える段なし ${a:.3f}、考える段込み ${b:.3f}（考える段 中央値 {think} トークン）")
            bj = bl = 0
            for r in rows:
                for k, g in r["gold"].items():
                    j = jev[r["id"]]["result"]["predicted"].get(k) == g
                    l_ = r["result"]["predicted"].get(k) == g
                    bj += j and not l_
                    bl += l_ and not j
            print(f"  - 条項ごとに Jev だけ正解 {bj} / LLM だけ正解 {bl}、p={mcnemar(bj, bl):.4f}")
        if mode in ("decisions", "llm-haiku", "llm-sonnet"):
            # 同じ契約書の条項は独立でないので、契約書ごとの正解数の差を単位にして検定する
            diffs = [sum(r["result"]["predicted"].get(k) == g for k, g in r["gold"].items())
                     - sum(jev[r["id"]]["result"]["predicted"].get(k) == g for k, g in r["gold"].items())
                     for r in rows if r["id"] in jev]
            m, p = sign_flip_p(diffs)
            print(f"  - 契約書単位の検定（Jev と {mode}、{len(diffs)} 通）: 正解数が違った契約書 {m} 通、p={p:.4f}")


goemotions()
goemotions_tuned()
goemotions_tuned("decisions")
contract()
