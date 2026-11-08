"""【計測用】実験4：検索結果を並べ替える（re-ranking）。

    uv run --python 3.12 --with rank-bm25 --with sentence-transformers --with typesafe-sdk scripts/recipe-rerank.py <方式>

方式: bm25 / embed / jev / jev-time / llm:haiku / llm-time:haiku / report

llm-time:<model> は、jev-time と同じ 30 問を llm:<model> と同じ方法で解き、API の時間を記録する。
jev-time は、等間隔に抜いた 30 問だけを jev と同じ方法で採点し、1 問の全段落を並列に採点し終えるまでの時間を記録する。

データは SQuAD 2.0 の dev（Stanford、CC BY-SA 4.0）。Wikipedia の記事 35 本の段落に、人が質問を付けた。
答えのある質問から等間隔に 300 問を抜き、質問ごとに「同じ記事の全段落（平均 37）」から、答えの入った段落を探させる。

- bm25: 単語の重なりで並べる。よくある速い検索
- embed: multilingual-e5-small の埋め込みの近さで並べる。手元の Mac で計算する
- jev: 公式のクックブック（rerank_typesafe）と同じく、質問と段落 1 つを 1 リクエストにして Noul を聞き、その値で並べる
- llm:<model>: 記事の全段落に番号を付けて claude -p に渡し、答えのある段落の番号を 1 つ選ばせる（100 問）

SQuAD の質問は段落を見ながら書かれたので、段落と同じ単語を使いやすい。単語の重なりで探す bm25 に有利なデータである。

出力: experiments/recipes/squad/results-<方式>.jsonl
"""

import asyncio
import json
import os
import re
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from lib.decisions_api import create as create_decision, require_key as require_decisions_key
from lib.claude_cli import model_id, model_metadata

DIR = Path("experiments/recipes/squad")
N_QUESTIONS = 300
N_LLM = 300  # 2026-11-05、Jev・Decisions APIと同じ300問にそろえた（以前は100問）
JEV_USD_PER_INPUT_TOKEN = 0.042 / 1_000_000


def load():
    articles = json.load(open(DIR / "data/dev-v2.0.json"))["data"]
    questions = []
    for a_i, article in enumerate(articles):
        for p_i, para in enumerate(article["paragraphs"]):
            for qa in para["qas"]:
                if not qa["is_impossible"]:
                    questions.append({"id": qa["id"], "article": a_i, "gold": p_i, "question": qa["question"]})
    step = len(questions) / N_QUESTIONS
    picked = [questions[int(k * step)] for k in range(N_QUESTIONS)]
    paragraphs = [[p["context"] for p in a["paragraphs"]] for a in articles]
    return picked, paragraphs


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def write(mode, rows):
    path = DIR / f"results-{mode.replace(':', '-')}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path}: {len(rows)} 問")


def run_bm25(questions, paragraphs):
    from rank_bm25 import BM25Okapi

    rows = []
    for q in questions:
        paras = paragraphs[q["article"]]
        scores = BM25Okapi([tokenize(p) for p in paras]).get_scores(tokenize(q["question"]))
        rows.append({**q, "scores": list(map(float, scores))})
    write("bm25", rows)


def run_embed(questions, paragraphs):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("intfloat/multilingual-e5-small")
    cache = {}
    rows = []
    started = time.time()
    for q in questions:
        if q["article"] not in cache:
            cache[q["article"]] = model.encode([f"passage: {p}" for p in paragraphs[q["article"]]], normalize_embeddings=True)
        e = model.encode([f"query: {q['question']}"], normalize_embeddings=True)[0]
        rows.append({**q, "scores": [float(x) for x in cache[q["article"]] @ e]})
    print(f"埋め込み {time.time() - started:.0f} 秒")
    write("embed", rows)


INSTRUCTIONS = "Does this Wikipedia paragraph contain the answer to the question?"
CRITERIA = {
    "true": "The paragraph states the fact the question asks for.",
    "false": "The paragraph is on a related topic but does not state the answer.",
}


async def run_jev(questions, paragraphs, mode="jev"):
    from typesafe_sdk import AsyncTypeSafeClient, Noul

    sem = asyncio.Semaphore(16)
    tokens = 0
    latencies = []

    async with AsyncTypeSafeClient() as client:

        async def score(question, paragraph):
            nonlocal tokens
            async with sem:
                t0 = time.time()
                r = await client.system_one(
                    state={"question": question, "paragraph": paragraph},
                    questions={"has_answer": Noul(instructions=INSTRUCTIONS, criteria=CRITERIA)},
                )
                latencies.append((time.time() - t0) * 1000)
                tokens += r.usage.input_tokens
                return r.answers["has_answer"].noul

        started = time.time()
        rows = []
        for q in questions:
            t_q = time.time()
            scores = await asyncio.gather(*(score(q["question"], p) for p in paragraphs[q["article"]]))
            rows.append({**q, "scores": scores, "wall_ms": round((time.time() - t_q) * 1000)})
        latencies.sort()
        print(f"{len(latencies)} 回、{time.time() - started:.0f} 秒、待ち 中央値 {latencies[len(latencies) // 2]:.0f}ms、${tokens * JEV_USD_PER_INPUT_TOKEN:.4f}")
    for r in rows:
        r["jev_input_tokens_total"] = tokens
    write(mode, rows)


def run_decisions(questions, paragraphs, mode="decisions"):
    """本に載せた一括方式と同様、質問を共有し、各段落を別の問いに置く。"""
    require_decisions_key()
    path = DIR / f"results-{mode}.jsonl"
    previous = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {row["id"] for row in previous if "scores" in row}
    with path.open("a") as stream:
        for index, case in enumerate(questions, 1):
            if case["id"] in done:
                continue
            paras = paragraphs[case["article"]]
            request_questions = [
                {"type": "predicate", "name": f"p{i}",
                 "instructions": (f"{INSTRUCTIONS}\nTrue: {CRITERIA['true']}\n"
                                  f"False: {CRITERIA['false']}\nParagraph: {paragraph}")}
                for i, paragraph in enumerate(paras)
            ]
            started = time.perf_counter()
            result = {}
            try:
                result = create_decision(json.dumps({"question": case["question"]}), request_questions)
                answers = result.pop("answers")
                scores = [answers[f"p{i}"]["probability"] for i in range(len(paras))]
                if any(not 0 <= score <= 1 for score in scores):
                    raise ValueError("invalid predicate probability")
                row = {**case, "scores": scores, "wall_ms": round((time.perf_counter() - started) * 1000), **result}
            except Exception as error:
                row = {**case, **getattr(error, "decisions_metadata", {}), **result, "error_type": type(error).__name__,
                       "wall_ms": round((time.perf_counter() - started) * 1000)}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(f"{index}/{len(questions)} {case['id']}: {'ok' if 'scores' in row else 'ERROR'}", flush=True)
            if "scores" not in row:
                raise RuntimeError("API呼び出しに失敗しました。保存結果を確認してから再開してください")


CHILD_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "TYPESAFE_"))}
# ユーザー階層の設定（~/.claude/CLAUDE.md など）を読ませない。scripts/lib/claude-cli.ts の ISOLATION_ARGS と同じ
ISOLATION_ARGS = ["--setting-sources", "project,local", "--strict-mcp-config"]
EMPTY_CWD = Path(tempfile.gettempdir()) / "jev-recipes-empty"
EMPTY_CWD.mkdir(exist_ok=True)


def decisions_requests(method, question, paras):
    """Jevの渡し方（recipe-rerank-batch.py の requests_for）と同じ形をDecisions APIで作る。

    条件は、どの実験とも同じく True:/False: の見出しを付けて質問文の後ろに並べる。
    """
    criteria = f"\nTrue: {CRITERIA['true']}\nFalse: {CRITERIA['false']}"
    if method == "separate":
        return [(json.dumps({"question": question, "paragraph": paragraph}),
                 [{"type": "predicate", "name": "has_answer", "instructions": INSTRUCTIONS + criteria}])
                for paragraph in paras]
    if method == "batch":
        return [(json.dumps({"question": question, "paragraphs": paras}),
                 [{"type": "predicate", "name": f"p{i}",
                   "instructions": (f"Does the Wikipedia paragraph at `paragraphs[{i}]` "
                                    f"contain the answer to `question`?{criteria}")}
                  for i in range(len(paras))])]
    raise ValueError(f"未知の方式: {method}")


def run_decisions_method(questions, paragraphs, method):
    """Jevで比べた残り2つの渡し方を、Decisions APIでも同じ300問で測る。通信の失敗だけ3回まで試し直す。"""
    from concurrent.futures import ThreadPoolExecutor
    require_decisions_key()
    path = DIR / f"results-decisions-{method}.jsonl"
    previous = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {row["id"] for row in previous if "scores" in row}

    def ask(request):
        for attempt in range(1, 4):
            try:
                result = create_decision(*request)
                return {**result, "attempts": attempt}
            except Exception as error:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)

    with path.open("a") as stream, ThreadPoolExecutor(8) as pool:
        for index, case in enumerate(questions, 1):
            if case["id"] in done:
                continue
            paras = paragraphs[case["article"]]
            requests = decisions_requests(method, case["question"], paras)
            started = time.perf_counter()
            try:
                results = list(pool.map(ask, requests))
                if method == "separate":
                    scores = [r["answers"]["has_answer"]["probability"] for r in results]
                else:
                    scores = [results[0]["answers"][f"p{i}"]["probability"] for i in range(len(paras))]
                row = {**case, "method": method, "scores": scores,
                       "wall_ms": round((time.perf_counter() - started) * 1000), "request_count": len(requests),
                       "model": results[0]["model"],
                       "input_tokens": sum(r["input_tokens"] for r in results),
                       "cost_usd": sum(r["cost_usd"] for r in results),
                       "attempts": sum(r["attempts"] for r in results),
                       "duration_ms": [round(r["duration_ms"], 1) for r in results]}
            except Exception as error:  # noqa: BLE001
                row = {**case, "method": method, "error_type": type(error).__name__,
                       "wall_ms": round((time.perf_counter() - started) * 1000)}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(f"{index}/{len(questions)} {case['id']}: {'ok' if 'scores' in row else 'ERROR'}", flush=True)
            if "scores" not in row:
                raise RuntimeError("API呼び出しに失敗しました。保存結果を確認してから再開してください")


def run_llm(questions, paragraphs, model, n=N_LLM, name=None):
    # 済んだ問（予測のある行）は残し、足りない問だけを測る
    path = DIR / f"results-{(name or f'llm:{model}').replace(':', '-')}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "predicted" in row:
                done[row["id"]] = row
    rows = []
    requested_model = model_id(model)
    for q in questions[:: N_QUESTIONS // n]:
        if q["id"] in done:
            rows.append(done[q["id"]])
            continue
        paras = paragraphs[q["article"]]
        schema = json.dumps(
            {"type": "object", "properties": {"paragraph": {"type": "integer", "minimum": 0, "maximum": len(paras) - 1}}, "required": ["paragraph"]}
        )
        body = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paras)) + f"\n\nQuestion: {q['question']}"
        out = subprocess.run(
            ["claude", "-p", "--model", requested_model, *ISOLATION_ARGS, "--tools", "", "--system-prompt",
             "You are given numbered Wikipedia paragraphs and a question. Answer with the number of the paragraph that contains the answer.",
             "--json-schema", schema, "--output-format", "json"],
            input=body, capture_output=True, text=True, env=CHILD_ENV, cwd=EMPTY_CWD,
        )
        try:
            d = json.loads(out.stdout)
            if d.get("stop_reason") == "refusal":
                # 回答拒否は通信失敗と分けて記録し、採点では不正解として数える
                rows.append({**q, "predicted": None, "refused": True, "result_text": str(d.get("result"))[:200],
                             **model_metadata(d, requested_model)})
                continue
            u = d["usage"]
            rows.append({**q, "predicted": d["structured_output"]["paragraph"],
                         "input_total": u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"],
                         "output_tokens": u["output_tokens"], "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
                         "duration_api_ms": d.get("duration_api_ms"),
                         **model_metadata(d, requested_model)})
        except Exception as e:  # noqa: BLE001
            rows.append({**q, "error": str(e)[:200]})
    write(name or f"llm:{model}", rows)


def expected_metrics(scores, gold):
    """同点の段落は、どの順にも等しい確率で並ぶとして期待値を出す。

    元の段落順で同点を並べると、確率を小数2桁で返すAPIほど段落の位置の影響を受ける。
    どの方式にも同じ数え方を当てるため、1位・5位以内・逆順位の期待値を返す。
    """
    gold_score = scores[gold]
    above = sum(score > gold_score for score in scores)
    tied = sum(score == gold_score for score in scores)
    top1 = 1 / tied if above == 0 else 0.0
    top5 = min(max((5 - above) / tied, 0.0), 1.0)
    rr = sum(1 / (above + j) for j in range(1, tied + 1)) / tied
    return top1, top5, rr


def paired_p(left, right, trials=200_000):
    """対応のある2方式の1位正解（期待値）の差の、符号反転による両側検定。乱数の種は固定。"""
    import random
    diffs = [a - b for a, b in zip(left, right) if abs(a - b) > 1e-12]
    observed = abs(sum(diffs))
    rng = random.Random(0)
    hits = sum(abs(sum(d if rng.random() < .5 else -d for d in diffs)) >= observed - 1e-12 for _ in range(trials))
    return len(diffs), hits / trials


def load_mode(mode):
    path = DIR / ("batch-20261029/results.jsonl" if mode.startswith("jev-batch") or mode == "jev-separate"
                  else f"results-{mode}.jsonl")
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if mode.startswith("jev-batch") or mode == "jev-separate":
        rows = [row for row in rows if row["method"] == mode[4:]]
    return rows


def report():
    print("同点の段落は、等しい確率でどの順にも並ぶとして期待値で数える。")
    print("| 方式 | 1 位が正解 | 5 位以内 | MRR |")
    top1 = {}
    llm_ids = None
    for mode in ["bm25", "embed", "jev-batch-local", "jev-batch", "jev-separate",
                 "decisions", "decisions-batch", "decisions-separate", "llm-haiku"]:
        rows = load_mode(mode)
        if rows is None:
            continue
        failures = sum("predicted" not in row if mode.startswith("llm")
                       else "scores" not in row or row["scores"] is None for row in rows)
        if failures:
            print(f"  {mode}: 対象 {len(rows)} 問、失敗 {failures} 問。比較採点を保留")
            continue
        if mode.startswith("llm"):
            llm_ids = {r["id"] for r in rows}
            top1[mode] = {r["id"]: float(r["predicted"] == r["gold"]) for r in rows}
            refused = sum(bool(r.get("refused")) for r in rows)
            print(f"| {mode}（{len(rows)} 問） | {st.mean(top1[mode].values()):.1%} | - | - |（回答拒否 {refused} 問を不正解として数えた）")
            continue
        if not rows:
            continue
        metrics = {r["id"]: expected_metrics(r["scores"], r["gold"]) for r in rows}
        top1[mode] = {id: m[0] for id, m in metrics.items()}
        tied = sum(sum(score == r["scores"][r["gold"]] for score in r["scores"]) > 1 for r in rows)
        print(f"| {mode}（{len(rows)} 問） | {st.mean(m[0] for m in metrics.values()):.1%} "
              f"| {st.mean(m[1] for m in metrics.values()):.1%} | {st.mean(m[2] for m in metrics.values()):.3f} |"
              f"（正解の段落が同点だった問 {tied}）")
    if llm_ids:
        for mode in ["bm25", "embed", "jev-batch-local", "decisions"]:
            if mode not in top1:
                continue
            sub = [top1[mode][id] for id in top1[mode] if id in llm_ids]
            if sub:
                print(f"  {mode} の LLM と同じ {len(sub)} 問: 1 位が正解 {st.mean(sub):.1%}")
    pairs = [("jev-batch-local", "decisions", None), ("jev-batch-local", "bm25", None),
             ("jev-batch-local", "llm-haiku", llm_ids), ("decisions", "llm-haiku", llm_ids)]
    for a, b, only in pairs:
        if a not in top1 or b not in top1 or (only is None and b.startswith("llm")):
            continue
        ids = [id for id in top1[a] if id in top1[b] and (only is None or id in only)]
        if not ids:
            continue
        n, p = paired_p([top1[a][id] for id in ids], [top1[b][id] for id in ids])
        print(f"  検定 {a} と {b}（{len(ids)} 問）: 差のある問 {n}、p={p:.4f}")


def main():
    mode = sys.argv[1]
    questions, paragraphs = load()
    if mode == "bm25":
        run_bm25(questions, paragraphs)
    elif mode == "embed":
        run_embed(questions, paragraphs)
    elif mode == "jev":
        asyncio.run(run_jev(questions, paragraphs))
    elif mode == "jev-time":
        asyncio.run(run_jev(questions[:: N_QUESTIONS // 30], paragraphs, "jev-time"))
    elif mode == "decisions":
        run_decisions(questions, paragraphs)
    elif mode in ("decisions-separate", "decisions-batch"):
        run_decisions_method(questions, paragraphs, mode.split("-", 1)[1])
    elif mode == "decisions-time":
        run_decisions(questions[:: N_QUESTIONS // 30], paragraphs, "decisions-time")
    elif mode.startswith("llm-time:"):
        run_llm(questions, paragraphs, mode[9:], 30, mode)
    elif mode.startswith("llm:"):
        run_llm(questions, paragraphs, mode[4:])
    elif mode == "report":
        report()


if __name__ == "__main__":
    main()
