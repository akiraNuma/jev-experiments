"""APIを呼ばず、まとめ送りの対応関係と測定記録の保護を検証する。"""
import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("batch", Path(__file__).with_name("recipe-rerank-batch.py"))
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def system_one(self, state, questions, model, timeout):
        self.calls.append({"state": state, "questions": questions, "model": model})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(.001)
        self.active -= 1
        values = {}
        for id, question in reversed(list(questions.items())):
            if "paragraph" in state:
                paragraph = state["paragraph"]
            elif "paragraphs" in state:
                index = int(id[1:])
                if f"`paragraphs[{index}]`" not in question.instructions:
                    raise AssertionError("質問IDだけで対象を示してはいけない")
                paragraph = state["paragraphs"][index]
            else:
                paragraph = question.instructions["paragraph"]
            values[id] = SimpleNamespace(noul=.95 if paragraph == "Closed on Monday." else .05)
        return SimpleNamespace(model=model, answers=values,
                               usage=SimpleNamespace(input_tokens=20, output_tokens=2))


class BatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_same_paragraph_order_despite_answer_map_order(self):
        for method in ["batch", "batch-local"]:
            with self.subTest(method=method):
                client = FakeClient()
                row = await batch.measure_case(client, {"id": "q", "article": 0, "gold": 1, "question": "Closed when?"},
                                               ["Near the station.", "Closed on Monday.", "Ten books."],
                                               method, batch.MODEL, 16)
                self.assertTrue(row["ok"])
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(row["scores"], [.05, .95, .05])
                self.assertEqual(max(range(3), key=lambda i: row["scores"][i]), 1)
                self.assertEqual(client.calls[0]["model"], batch.MODEL)

    async def test_separate_requests_have_parallel_limit(self):
        client = FakeClient()
        row = await batch.measure_case(client, {"id": "q", "article": 0, "gold": 0, "question": "Where?"},
                                       ["Paragraph"] * 35, "separate", batch.MODEL, 16)
        self.assertTrue(row["ok"])
        self.assertEqual(row["request_count"], 35)
        self.assertEqual(len(row["scores"]), 35)
        self.assertLessEqual(client.max_active, 16)
        self.assertGreater(client.max_active, 1)

    async def test_failure_does_not_invent_usage(self):
        class FailedClient:
            async def system_one(self, **kwargs):
                raise ConnectionError("not logged")
        row = await batch.measure_case(FailedClient(), {"id": "q", "article": 0, "gold": 0, "question": "Where?"},
                                       ["Paragraph"], "batch", batch.MODEL, 16)
        self.assertFalse(row["ok"])
        self.assertIsNone(row["cost_usd"])
        self.assertIsNone(row["input_tokens"])
        self.assertEqual(row["requests"][0]["error_type"], "ConnectionError")
        self.assertNotIn("not logged", json.dumps(row))

    async def test_validation_failure_preserves_received_usage(self):
        class WrongModelClient:
            async def system_one(self, **kwargs):
                return SimpleNamespace(model="other", answers={},
                                       usage=SimpleNamespace(input_tokens=100, output_tokens=2))
        row = await batch.measure_case(WrongModelClient(),
                                       {"id": "q", "article": 0, "gold": 0, "question": "Where?"},
                                       ["Paragraph"], "batch", batch.MODEL, 16)
        self.assertFalse(row["ok"])
        self.assertIsNone(row["cost_usd"])
        self.assertAlmostEqual(row["known_cost_usd"], 100 * batch.PRICE)
        self.assertEqual(row["requests"][0]["input_tokens"], 100)

    def test_bad_response_is_rejected(self):
        for response in [
            SimpleNamespace(model="other", answers={"p0": SimpleNamespace(noul=.5)}),
            SimpleNamespace(model=batch.MODEL, answers={}),
            SimpleNamespace(model=batch.MODEL, answers={"p0": SimpleNamespace(noul=1.5)}),
        ]:
            with self.assertRaises(ValueError):
                batch.read_scores(response, ["p0"], batch.MODEL)

    def test_resume_rejects_changed_conditions_without_overwriting(self):
        config = {"model": batch.MODEL, "ids": ["q"], "methods": ["batch"], "request_sha256": "a"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertEqual(batch.open_measurement(path, config), [])
            row = {"id": "q", "method": "batch", "ok": True}
            result_path = path / "results.jsonl"
            result_path.write_text(json.dumps(row) + "\n")
            self.assertEqual(batch.open_measurement(path, config), [row])
            with self.assertRaises(ValueError):
                batch.open_measurement(path, {**config, "model": "other"})
            self.assertEqual(result_path.read_text(), json.dumps(row) + "\n")

    def test_resume_rejects_unrelated_result(self):
        config = {"model": batch.MODEL, "ids": ["q"], "methods": ["batch"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            batch.open_measurement(path, config)
            (path / "results.jsonl").write_text(json.dumps({"id": "not-q", "method": "batch"}) + "\n")
            with self.assertRaises(ValueError):
                batch.open_measurement(path, config)


if __name__ == "__main__":
    unittest.main()
