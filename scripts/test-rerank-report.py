"""掲載方式の参照と、LLMの回答形式を混同しないことを既知の採点で検証する。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("rerank", Path(__file__).with_name("recipe-rerank.py"))
rerank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rerank)


class RerankReportTest(unittest.TestCase):
    def test_uses_published_batch_local_and_accepts_llm_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "batch-20261029").mkdir()
            def write(name, rows):
                (root / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            cases = [{"id": "a", "gold": 1}, {"id": "b", "gold": 0}]
            write("results-jev.jsonl", [{**row, "scores": [0, 0]} for row in cases])
            write("batch-20261029/results.jsonl", [
                {**cases[0], "method": "batch-local", "scores": [.1, .9]},
                {**cases[1], "method": "batch-local", "scores": [.9, .1]},
                {**cases[0], "method": "batch", "scores": [.9, .1]},
            ])
            write("results-llm-haiku.jsonl", [{**row, "predicted": row["gold"]} for row in cases])
            output = io.StringIO()
            with patch.object(rerank, "DIR", root), contextlib.redirect_stdout(output):
                rerank.report()
            text = output.getvalue()
            self.assertIn("jev-batch-local（2 問） | 100.0%", text)
            self.assertIn("llm-haiku（2 問） | 100.0%", text)
            self.assertIn("jev-batch-local の LLM と同じ 2 問", text)
            self.assertNotIn("失敗", text)

    def test_failed_ranking_is_reported_before_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results-decisions.jsonl").write_text(json.dumps({"id": "a", "error_type": "ValueError"}) + "\n")
            output = io.StringIO()
            with patch.object(rerank, "DIR", root), contextlib.redirect_stdout(output):
                rerank.report()
            self.assertIn("対象 1 問、失敗 1 問。比較採点を保留", output.getvalue())


if __name__ == "__main__":
    unittest.main()
