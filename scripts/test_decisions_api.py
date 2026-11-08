"""ネットワークを使わず、Python側の送信・利用量検査を確かめる。"""

import copy
import hashlib
import json
import os
import unittest
from unittest.mock import patch

from lib import decisions_api


class FakeResponse:
    status = 200

    def read(self):
        return json.dumps({
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": "has_answer", "probability": 0.8}],
            "usage": {"input_tokens": 2000,
                      "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 500}},
        }).encode()


class FakeConnection:
    def __init__(self, *_args, **_kwargs):
        self.body = None

    def request(self, _method, _path, body, headers):
        self.body = json.loads(body)
        assert headers["Authorization"] == "Bearer test-key"

    def getresponse(self):
        return FakeResponse()

    def close(self):
        pass


class DecisionsApiTest(unittest.TestCase):
    def call_payload(self, payload, question=None):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch.object(decisions_api.http.client, "HTTPSConnection", FakeConnection), \
             patch.object(FakeResponse, "read", return_value=json.dumps(payload).encode()):
            decisions_api._LOCAL.connection = None
            try:
                return decisions_api.create("question", [question or {
                    "type": "predicate", "name": "has_answer", "instructions": "Does it?"}])
            finally:
                decisions_api._LOCAL.connection = None

    def test_invalid_predicate_and_usage(self):
        valid = json.loads(FakeResponse().read())
        bad = []
        for value in (-0.1, 1.1, True, "0.8", None):
            payload = copy.deepcopy(valid)
            payload["answers"][0]["probability"] = value
            bad.append(payload)
        for usage in ({"input_tokens": True}, {"input_tokens": 1.5},
                      {"input_tokens": 2, "input_tokens_details": {"cached_tokens": "oops"}},
                      {"input_tokens": 2, "input_tokens_details": {"cache_write_tokens": 3}}):
            bad.append({**valid, "usage": usage})
        bad.extend([{**valid, "model": "wrong-model"}, {**valid, "answers": []},
                    {**valid, "answers": [{"type": "choice", "name": "has_answer"}]}])
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.call_payload(payload)

    def test_choice_distribution_and_refusal(self):
        payload = json.loads(FakeResponse().read())
        question = {"type": "choice", "name": "route", "instructions": "route it",
                    "choices": [{"value": "a"}, {"value": "b"}]}
        answer = {"type": "choice", "name": "route", "choice": "a", "confidence": .9,
                  "probabilities": [{"value": "a", "probability": .9}, {"value": "b", "probability": .1}]}
        payload["answers"] = [answer]
        self.assertEqual(self.call_payload(payload, question)["answers"]["route"]["choice"], "a")
        for changes in ({"confidence": 2}, {"choice": "c"}, {"probabilities": []},
                        {"probabilities": [{"value": "a", "probability": .5}] * 2},
                        {"probabilities": [{"value": "a", "probability": .1}, {"value": "b", "probability": .1}]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call_payload({**payload, "answers": [{**answer, **changes}]}, question)
        payload["answers"] = [{"type": "refusal", "name": "route"}]
        self.assertEqual(self.call_payload(payload, question)["answers"]["route"]["type"], "refusal")

    def test_score_range(self):
        payload = json.loads(FakeResponse().read())
        question = {"type": "score", "name": "severity", "instructions": "rate it",
                    "levels": [{"label": "low"}, {"label": "high"}]}
        answer = {"type": "score", "name": "severity", "score": .8, "confidence": .5,
                  "probabilities": [{"value": 0, "label": "low", "probability": .2},
                                    {"value": 1, "label": "high", "probability": .8}]}
        self.assertEqual(self.call_payload({**payload, "answers": [answer]}, question)["answers"]["severity"]["score"], .8)
        for value in (2, 0.2):
            with self.assertRaises(ValueError) as context:
                self.call_payload({**payload, "answers": [{**answer, "score": value}]}, question)
            self.assertRegex(context.exception.decisions_metadata["request_sha256"], r"^[a-f0-9]{64}$")

    def test_request_and_usage(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch.object(decisions_api.http.client, "HTTPSConnection", FakeConnection):
            decisions_api._LOCAL.connection = None
            result = decisions_api.create("question", [
                {"type": "predicate", "name": "has_answer", "instructions": "Does it?"},
            ])
            self.assertEqual(result["answers"]["has_answer"]["probability"], 0.8)
            self.assertEqual(result["input_tokens"], 2000)
            self.assertEqual(result["request_sha256"], hashlib.sha256(json.dumps(decisions_api._LOCAL.connection.body).encode()).hexdigest())
            self.assertEqual(result["cost_usd"], (500 * .10 + 1000 * .01 + 500 * .125) / 1_000_000)
            self.assertEqual(decisions_api._LOCAL.connection.body["model"], "gpt-6-luna")
            decisions_api._LOCAL.connection = None


if __name__ == "__main__":
    unittest.main()
