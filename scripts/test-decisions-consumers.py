"""架空データと模擬HTTPで、各測定スクリプトの拒否記録を末端まで検証する。"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RefusalRecordingTest(unittest.TestCase):
    def test_all_measurement_paths_keep_received_usage_and_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / "node_modules").symlink_to(ROOT / "node_modules", target_is_directory=True)
            mock = work / "mock.mjs"
            mock.write_text('''globalThis.fetch = async (url, options) => {
  if (url !== "https://api.openai.com/v1/decisions") throw new Error("Unexpected network call");
  const body = JSON.parse(options.body);
  return new Response(JSON.stringify({model: "gpt-6-luna",
    answers: body.questions.map(q => ({type: "refusal", name: q.name})),
    usage: {input_tokens: 100, input_tokens_details: {cached_tokens: 0, cache_write_tokens: 0}}
  }), {status: 200});
};''')

            def write(name, value, raw=False):
                path = work / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value if raw else json.dumps(value))

            write("experiments/classify/data/banking77/test.csv", "text,category\nhello,other\n", True)
            write("experiments/classify/data/banking77/categories.json", ["other", "billing"])
            write("experiments/recipes/goemotions/data/emotions.txt", "joy\nneutral\n", True)
            write("experiments/recipes/goemotions/data/test.tsv", "hello\t0\tcase\n", True)
            write("experiments/recipes/toxicity-ja/data/toxicity_dataset.jsonl",
                  {"id": "case", "text": "hello", "label": "nontoxic"})
            qa = {"question": "Which day?", "knowledge": "Monday", "right_answer": "Monday", "hallucinated_answer": "Friday"}
            write("experiments/recipes/halueval/data/qa_data.json", json.dumps(qa) + "\n" + json.dumps(qa), True)
            write("experiments/recipes/contractnli/data/test.json", {
                "labels": {"clause": {"hypothesis": "Keep this confidential."}},
                "documents": [{"id": 1, "text": "Keep this confidential.",
                               "annotation_sets": [{"annotations": {"clause": {"choice": "Entailment"}}}]}]})
            write("experiments/recipes/cuad/data/agreement-date-test.json", [{"title": "case",
                  "context": "Signed on March 1, 2020.", "answers": [{"text": "March 1, 2020", "answer_start": 10}]}])
            write("experiments/recipes/pets/manifest.json", {"items": [{"id": "case", "breed": "British_Shorthair",
                  "species": "cat", "split": "eval", "sha256": "synthetic", "box_784": {}}]})
            write("experiments/recipes/pets/data/case-384.jpg", "synthetic bytes; HTTP is mocked", True)
            cases = [
                ("classify-judge.ts", ["banking77", "decisions", "--sample", "1"], "classify/results-banking77-decisions.jsonl"),
                ("recipe-multilabel.ts", ["decisions", "--sample", "1"], "recipes/goemotions/results-decisions.jsonl"),
                ("recipe-toxicity.ts", ["decisions", "--sample", "1"], "recipes/toxicity-ja/results-decisions.jsonl"),
                ("recipe-halueval.ts", ["decisions", "--sample", "2"], "recipes/halueval/results-decisions.jsonl"),
                ("recipe-contract.ts", ["decisions"], "recipes/contractnli/results-decisions.jsonl"),
                ("recipe-extract.ts", ["decisions"], "recipes/cuad/results-decisions.jsonl"),
                ("recipe-images.ts", ["decisions-text", "eval"], "recipes/pets/results-decisions-text-eval.jsonl"),
            ]
            for script, args, output in cases:
                with self.subTest(script=script):
                    completed = subprocess.run(["node", "--import", "tsx", "--import", str(mock),
                        str(ROOT / "scripts" / script), *args], cwd=work, capture_output=True, text=True,
                        env={**os.environ, "OPENAI_API_KEY": "test-key", "TYPESAFE_API_KEY": "test-key"}, timeout=30)
                    path = work / "experiments" / output
                    self.assertTrue(path.exists(), completed.stderr)
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertGreater(len(rows), 0)
                    for row in rows:
                        result = row["result"]
                        self.assertFalse(result["ok"])
                        self.assertEqual(result["model"], "gpt-6-luna")
                        self.assertEqual(result["input_tokens"], 100)
                        self.assertAlmostEqual(result["cost_usd"], .00001)
                        self.assertRegex(result["request_sha256"], r"^[a-f0-9]{64}$")
                        self.assertTrue(result["refused_questions"])
                        self.assertTrue(all(answer["type"] == "refusal" for answer in result["answers"].values()))


if __name__ == "__main__":
    unittest.main()
