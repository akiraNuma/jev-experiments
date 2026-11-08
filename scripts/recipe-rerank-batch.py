"""実験4の再計測。段落ごとの送信と、複数のNoulをまとめる送信を比べる。

uv run --python 3.12 --with typesafe-sdk==0.6.0 scripts/recipe-rerank-batch.py plan --output experiments/recipes/squad/batch-20261029
同じ引数のplanをrunへ変えるとAPIを呼ぶ。reportは保存結果だけを集計する。
.envは読まない。TYPESAFE_API_KEYは実行環境からSDKが読む。
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import msgspec
from typesafe_sdk import AsyncTypeSafeClient, Noul

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments/recipes/squad"
MODEL = "jev-1.13.0"
PRICE = 0.042 / 1_000_000
METHODS = ("separate", "batch", "batch-local")
INSTRUCTIONS = "Does this Wikipedia paragraph contain the answer to the question?"
CRITERIA = {
    "true": "The paragraph states the fact the question asks for.",
    "false": "The paragraph is on a related topic but does not state the answer.",
}


def load_cases(limit):
    spec = importlib.util.spec_from_file_location("rerank_source", ROOT / "scripts/recipe-rerank.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    source.DIR = DATA
    questions, paragraphs = source.load()
    return questions[:limit], paragraphs


def requests_for(method, question, paragraphs):
    if method == "separate":
        return [
            {
                "state": {"question": question, "paragraph": paragraph},
                "questions": {"has_answer": Noul(instructions=INSTRUCTIONS, criteria=CRITERIA)},
            }
            for paragraph in paragraphs
        ]
    if method == "batch":
        return [{
            "state": {"question": question, "paragraphs": paragraphs},
            "questions": {
                f"p{i}": Noul(
                    instructions=(
                        f"Does the Wikipedia paragraph at `paragraphs[{i}]` "
                        "contain the answer to `question`?"
                    ),
                    criteria=CRITERIA,
                )
                for i in range(len(paragraphs))
            },
        }]
    if method == "batch-local":
        return [{
            "state": {"question": question},
            "questions": {
                f"p{i}": Noul(
                    instructions={"instruction": INSTRUCTIONS, "paragraph": paragraph},
                    criteria=CRITERIA,
                )
                for i, paragraph in enumerate(paragraphs)
            },
        }]
    raise ValueError(f"未知の方式: {method}")


def fingerprint(cases, paragraphs, methods):
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case["id"].encode())
        digest.update(str(case["gold"]).encode())
        for method in methods:
            for request in requests_for(method, case["question"], paragraphs[case["article"]]):
                digest.update(msgspec.json.encode(request))
    return digest.hexdigest()


def open_measurement(output, config):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["config"] != config:
            raise ValueError("保存済みの測定と条件が異なります。別の出力先を指定してください")
    else:
        if (output / "results.jsonl").exists():
            raise ValueError("manifestのない結果には追記しません")
        with manifest.open("x") as stream:
            json.dump({"created_at": datetime.now(timezone.utc).isoformat(), "config": config},
                      stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    rows = []
    if (output / "results.jsonl").exists():
        for number, line in enumerate((output / "results.jsonl").read_text().splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"保存結果の{number}行目が不正です。追記を中止します") from error
            row = rows[-1]
            if row.get("id") not in config["ids"] or row.get("method") not in config["methods"]:
                raise ValueError(f"保存結果の{number}行目に対象外の質問・方式があります")
    return rows


def read_scores(response, ids, model):
    if response.model != model:
        raise ValueError("応答したモデルが指定と異なります")
    if set(response.answers) != set(ids):
        raise ValueError("返った判定のIDがリクエストと異なります")
    values = [response.answers[id].noul for id in ids]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("確率が0〜1の範囲にありません")
    return values


async def measure_case(client, case, paragraphs, method, model, parallel):
    requests = requests_for(method, case["question"], paragraphs)
    semaphore = asyncio.Semaphore(parallel)

    async def ask(index, request):
        async with semaphore:
            started = time.perf_counter()
            usage = {}
            try:
                response = await client.system_one(**request, model=model, timeout=60)
                if (not isinstance(response.usage.input_tokens, int) or response.usage.input_tokens < 0
                        or not isinstance(response.usage.output_tokens, int) or response.usage.output_tokens < 0):
                    raise ValueError("利用トークン数を取得できませんでした")
                usage = {"input_tokens": response.usage.input_tokens,
                         "output_tokens": response.usage.output_tokens}
                values = read_scores(response, list(request["questions"]), model)
                return {
                    "ok": True, "index": index, "scores": values, "model": response.model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            except Exception as error:
                # 例外の本文・HTTPヘッダー・入力文章は保存や表示をしない。
                return {
                    "ok": False, "index": index, "error_type": type(error).__name__,
                    "status_code": getattr(error, "status_code", None),
                    **usage,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }

    started = time.perf_counter()
    # 失敗時も、すでに始めた呼び出しが終わるまで待って記録する。
    responses = await asyncio.gather(*(ask(i, request) for i, request in enumerate(requests)))
    wall_ms = round((time.perf_counter() - started) * 1000, 3)
    ok = all(response["ok"] for response in responses)
    scores = [value for response in responses for value in response.get("scores", [])] if ok else None
    input_tokens = sum(response.get("input_tokens") or 0 for response in responses)
    output_tokens = sum(response.get("output_tokens") or 0 for response in responses)
    return {
        "id": case["id"], "article": case["article"], "gold": case["gold"],
        "method": method, "ok": ok, "scores": scores, "paragraphs": len(paragraphs),
        "wall_ms": wall_ms, "request_count": len(requests),
        "input_tokens": input_tokens if ok else None, "output_tokens": output_tokens if ok else None,
        "cost_usd": input_tokens * PRICE if ok else None,
        "known_cost_usd": input_tokens * PRICE, "requests": responses,
    }


def summarize(output, rows):
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["config"]
    latest = {(row["id"], row["method"]): row for row in rows}
    summary = {
        "model": config["model"], "sdk_version": config["sdk_version"],
        "expected_questions_per_method": len(config["ids"]),
        "recorded_cost_usd_all_attempts": sum(row["known_cost_usd"] for row in rows),
        "failed_attempts_with_incomplete_usage": sum(
            any("input_tokens" not in request for request in row["requests"]) for row in rows),
        "methods": {},
    }
    haiku = {row["id"]: row for row in map(json.loads, (DATA / "results-llm-haiku.jsonl").read_text().splitlines())
             if "predicted" in row}
    for method in config["methods"]:
        selected = [latest[(id, method)] for id in config["ids"]
                    if (id, method) in latest and latest[(id, method)]["ok"]]
        if not selected:
            summary["methods"][method] = {"complete": False, "n": 0}
            continue
        correct = lambda row: max(range(len(row["scores"])), key=lambda i: row["scores"][i]) == row["gold"]
        paired = [row for row in selected if row["id"] in haiku]
        thirty = [row for row in selected if row["id"] in config["timing_ids"]]
        summary["methods"][method] = {
            "complete": len(selected) == len(config["ids"]), "n": len(selected),
            "correct": sum(correct(row) for row in selected),
            "accuracy": sum(correct(row) for row in selected) / len(selected),
            "cost_per_question": statistics.mean(row["cost_usd"] for row in selected),
            "input_tokens_per_question": statistics.mean(row["input_tokens"] for row in selected),
            "requests_total": sum(row["request_count"] for row in selected),
            "wall_ms_median": statistics.median(row["wall_ms"] for row in selected),
            "wall_ms_max": max(row["wall_ms"] for row in selected),
            "historical_30_subset": {
                "n": len(thirty),
                "wall_ms_median": statistics.median(row["wall_ms"] for row in thirty) if thirty else None,
            },
            "historical_haiku_subset": {
                "n": len(paired), "correct": sum(correct(row) for row in paired),
                "haiku_correct": sum(haiku[row["id"]]["predicted"] == row["gold"] for row in paired),
                "only_this_method_correct": sum(correct(row) and haiku[row["id"]]["predicted"] != row["gold"] for row in paired),
                "only_haiku_correct": sum(not correct(row) and haiku[row["id"]]["predicted"] == row["gold"] for row in paired),
            },
        }
    (output / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


async def run(output, cases, paragraphs, methods, model, parallel, budget, rows):
    done = {(row["id"], row["method"]) for row in rows if row["ok"]}
    cost = sum(row["known_cost_usd"] for row in rows)
    async with AsyncTypeSafeClient() as client:
        for index, case in enumerate(cases):
            # 呼ぶ順番を交互にして、どちらかが常に先にならないようにする。
            order = methods if index % 2 == 0 else list(reversed(methods))
            for method in order:
                if (case["id"], method) in done:
                    continue
                if cost >= budget:
                    raise RuntimeError("記録した利用量が設定予算に達しました。結果は保存済みです")
                row = await measure_case(client, case, paragraphs[case["article"]], method, model, parallel)
                with (output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                cost += row["known_cost_usd"]
                print(f"{index + 1}/{len(cases)} {method}: ok={row['ok']} "
                      f"requests={row['request_count']} wall_ms={row['wall_ms']:.0f} "
                      f"recorded_cost=${cost:.5f}", flush=True)
                if not row["ok"]:
                    summarize(output, rows)
                    raise RuntimeError("API呼び出しに失敗しました。保存結果を確認してから再開してください")
    return summarize(output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run", "report"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", default="separate,batch,batch-local")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--budget-usd", type=float, default=1.0)
    args = parser.parse_args()
    if args.action == "report":
        rows = [json.loads(line) for line in (args.output / "results.jsonl").read_text().splitlines()]
        print(json.dumps(summarize(args.output, rows), ensure_ascii=False))
        return
    methods = args.methods.split(",")
    if (not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods)
            or not 1 <= args.limit <= 300 or args.parallel < 1 or not math.isfinite(args.budget_usd) or args.budget_usd <= 0):
        parser.error("方式・件数・並列数・予算の指定が不正です")
    cases, paragraphs = load_cases(args.limit)
    config = {
        "methods": methods, "model": args.model, "parallel": args.parallel,
        "sdk_version": importlib.metadata.version("typesafe-sdk"),
        "dataset_sha256": hashlib.sha256((DATA / "data/dev-v2.0.json").read_bytes()).hexdigest(),
        "request_sha256": fingerprint(cases, paragraphs, methods),
        "ids": [case["id"] for case in cases], "timing_ids": [case["id"] for case in cases[::10]],
    }
    rows = open_measurement(args.output, config)
    if args.action == "plan":
        print(json.dumps({"questions": len(cases), "methods": methods, "model": args.model,
                          "paragraphs_mean": statistics.mean(len(paragraphs[case["article"]]) for case in cases),
                          "separate_requests": sum(len(paragraphs[case["article"]]) for case in cases),
                          "batch_requests_per_method": len(cases)}, ensure_ascii=False))
        return
    try:
        result = asyncio.run(run(args.output, cases, paragraphs, methods, args.model, args.parallel,
                                 args.budget_usd, rows))
    except Exception:
        summarize(args.output, rows)
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
