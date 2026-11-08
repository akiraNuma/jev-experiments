"""比較用のOpenAI Decisions API呼び出し。Jevの測定経路からは使わない。"""

import hashlib
import http.client
import json
import math
import os
import threading
import time

MODEL = "gpt-6-luna"
_LOCAL = threading.local()


def require_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured; set it in .env and run uv with --env-file .env")
    return key


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def create(input_value, questions):
    """再試行しない。失敗した呼び出しの課金は不明として扱う。"""
    key = require_key()
    connection = getattr(_LOCAL, "connection", None)
    if connection is None:
        connection = http.client.HTTPSConnection("api.openai.com", timeout=60)
        _LOCAL.connection = connection
    body = json.dumps({"model": MODEL, "input": input_value, "questions": questions})
    started = time.perf_counter()
    try:
        connection.request("POST", "/v1/decisions", body=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        })
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"Decisions API HTTP {response.status}")
        data = json.loads(raw)
        answers = data.get("answers")
        usage = data.get("usage", {})
        if data.get("model") != MODEL:
            raise ValueError("Decisions API returned an unexpected model")
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise ValueError("Decisions API returned missing answers")
        by_name = {answer.get("name"): answer for answer in answers}
        if set(by_name) != {question["name"] for question in questions}:
            raise ValueError("Decisions API returned unexpected answer names")
        for question in questions:
            answer = by_name[question["name"]]
            kind = answer.get("type")
            if kind == "refusal":
                continue
            if kind != question["type"]:
                raise ValueError("Decisions API returned a wrong answer type")
            if kind == "predicate":
                if not probability(answer.get("probability")):
                    raise ValueError("Decisions API returned an invalid probability")
                continue
            if not probability(answer.get("confidence")):
                raise ValueError("Decisions API returned an invalid confidence")
            expected = ([choice["value"] for choice in question["choices"]] if kind == "choice"
                        else list(range(len(question["levels"]))))
            distribution = answer.get("probabilities")
            if (not isinstance(distribution, list) or len(distribution) != len(expected)
                    or any(not isinstance(entry, dict) or not probability(entry.get("probability"))
                           for entry in distribution)):
                raise ValueError("Decisions API returned an invalid distribution")
            values = [entry.get("value") for entry in distribution]
            if (len(set(values)) != len(expected) or set(values) != set(expected)
                    or abs(sum(entry["probability"] for entry in distribution) - 1) > 0.001):
                raise ValueError("Decisions API returned invalid distribution options")
            if kind == "choice" and answer.get("choice") not in expected:
                raise ValueError("Decisions API returned an invalid choice")
            if kind == "score":
                score = answer.get("score")
                if (type(score) not in (int, float) or not math.isfinite(score)
                        or not 0 <= score <= len(expected) - 1
                        or abs(sum(entry["value"] * entry["probability"] for entry in distribution) - score) > 0.001
                        or any(type(entry["value"]) is not int or
                               entry.get("label") != question["levels"][entry["value"]]["label"]
                               for entry in distribution)):
                    raise ValueError("Decisions API returned an invalid score")
        tokens = usage.get("input_tokens")
        details = usage.get("input_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        written = details.get("cache_write_tokens", 0)
        if not all(type(value) is int and value >= 0 for value in (tokens, cached, written)) or cached + written > tokens:
            raise ValueError("Decisions API returned invalid token usage")
        # 2026-11-01の標準・短文脈料金。長文脈と地域処理の加算は含めない。
        cost = ((tokens - cached - written) * 0.10 + cached * 0.01 + written * 0.125) / 1_000_000
        return {"model": data.get("model"), "answers": by_name, "input_tokens": tokens,
                "request_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "refused_questions": [name for name, answer in by_name.items() if answer["type"] == "refusal"],
                "cached_input_tokens": cached, "cache_write_tokens": written,
                "cost_usd": cost, "duration_ms": (time.perf_counter() - started) * 1000}
    except Exception as error:
        error.decisions_metadata = {"request_sha256": hashlib.sha256(body.encode()).hexdigest(),
                                    "requested_model": MODEL,
                                    "duration_ms": (time.perf_counter() - started) * 1000}
        connection.close()
        _LOCAL.connection = None
        raise
