"""模擬CLIで、指定モデルと応答モデルを混同せず保存することを確認する。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import ast
import importlib.util
from types import SimpleNamespace
import time

ROOT = Path(__file__).resolve().parents[1]


class ClaudeMeasurementTest(unittest.TestCase):
    def test_overhead_outputs_keep_each_calibration_call_model(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / 'node_modules').symlink_to(ROOT / 'node_modules', target_is_directory=True)
            tool = work / 'claude'
            tool.write_text(f'#!{sys.executable}\n' + '''import json, sys
json.dump({'structured_output': {'a': 'a'}, 'total_cost_usd': .001,
 'usage': {'input_tokens': 10, 'cache_creation_input_tokens': 0,
           'cache_read_input_tokens': 0, 'output_tokens': 1},
 'modelUsage': {'claude-sonnet-5-5': {'inputTokens': 10}}}, sys.stdout)
''')
            tool.chmod(0o755)

            def write(name, value, raw=False):
                path = work / 'experiments/recipes' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value if raw else json.dumps(value))

            write('goemotions/data/emotions.txt', 'joy\nneutral\n', True)
            write('goemotions/data/test.tsv', 'hello\t0\tcase\n', True)
            write('toxicity-ja/data/toxicity_dataset.jsonl', {'id': 'case', 'text': 'hello', 'label': 'nontoxic'})
            write('contractnli/data/test.json', {'labels': {'clause': {'hypothesis': 'Keep this confidential.'}},
                'documents': [{'id': 1, 'text': 'Keep this confidential.',
                    'annotation_sets': [{'annotations': {'clause': {'choice': 'Entailment'}}}]}]})
            write('cuad/data/agreement-date-test.json', [{'title': 'case', 'context': 'Signed on March 1, 2020.',
                'answers': [{'text': 'March 1, 2020', 'answer_start': 10}]}])
            qa = {'question': 'Which day?', 'knowledge': 'Monday', 'right_answer': 'Monday', 'hallucinated_answer': 'Friday'}
            write('halueval/data/qa_data.json', json.dumps(qa)+'\n'+json.dumps(qa), True)
            for script, dataset in [('multilabel', 'goemotions'), ('toxicity', 'toxicity-ja'),
                                    ('contract', 'contractnli'), ('extract', 'cuad'), ('halueval', 'halueval')]:
                with self.subTest(script=script):
                    completed = subprocess.run(['node', '--import', 'tsx', str(ROOT/f'scripts/recipe-{script}.ts'),
                        'overhead:haiku'], cwd=work, capture_output=True, text=True,
                        env={**os.environ, 'PATH': str(work)+os.pathsep+os.environ['PATH'], 'TYPESAFE_API_KEY': 'test-key'}, timeout=30)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    row = json.loads((work/f'experiments/recipes/{dataset}/overhead-haiku.json').read_text())
                    self.assertEqual(row['tokens'], 10)
                    self.assertEqual(row['requested_model'], 'claude-haiku-5-5')
                    self.assertEqual(row['models_used'], ['claude-sonnet-5-5'])
                    self.assertEqual(len(row['samples']), 3)
                    for sample in row['samples']:
                        self.assertEqual(sample['model_usage'], {'claude-sonnet-5-5': {'inputTokens': 10}})

    def test_both_cli_paths_pin_request_and_keep_response_models(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / 'node_modules').symlink_to(ROOT / 'node_modules', target_is_directory=True)
            tool = work / 'claude'
            tool.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
Path(os.environ['CLI_TEST_ARGS']).write_text(json.dumps(sys.argv[1:]))
json.dump({'structured_output': {'intent': 'other', 'decision': 'allow'}, 'total_cost_usd': .001,
 'usage': {'input_tokens': 10, 'cache_creation_input_tokens': 0,
           'cache_read_input_tokens': 0, 'output_tokens': 1},
 'modelUsage': json.loads(os.environ['CLI_TEST_MODELS'])}, sys.stdout)
''')
            tool.chmod(0o755)
            data = work / 'experiments/classify/data/banking77'
            data.mkdir(parents=True)
            (data / 'test.csv').write_text('text,category\nhello,other\n')
            (data / 'categories.json').write_text('["other"]')
            cases = [('classify-judge.ts', 'llm:haiku', ['banking77', 'llm:haiku', '--sample', '1'],
                      'classify/results-banking77-llm-haiku.jsonl'),
                     ('classify-judge.ts', 'tokens:haiku', ['banking77', 'tokens:haiku', '--sample', '1'],
                      'classify/results-banking77-tokens-haiku.jsonl')]
            for script, mode, run_args, name in cases:
                for returned in ({'claude-sonnet-5-5': {'inputTokens': 10}}, {}):
                    with self.subTest(mode=mode, returned=returned):
                        # Use a new output name per run, without changing saved experiment records.
                        output = work / 'experiments' / name
                        if output.exists():
                            output.rename(output.with_suffix('.previous'))
                        result = subprocess.run(['node', '--import', 'tsx', str(ROOT / 'scripts' / script),
                            *run_args], cwd=work, capture_output=True, text=True,
                            env={**os.environ, 'PATH': str(work) + os.pathsep + os.environ['PATH'],
                                 'CLI_TEST_ARGS': str(work / 'args.json'), 'CLI_TEST_MODELS': json.dumps(returned),
                                 'TYPESAFE_API_KEY': 'test-key'}, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        row = json.loads(output.read_text())['result']
                        self.assertTrue(row['ok'])
                        self.assertEqual(row['requested_model'], 'claude-haiku-5-5')
                        self.assertEqual(row['models_used'], list(returned))
                        self.assertEqual(row['model_usage'], returned)
                        args = json.loads((work / 'args.json').read_text())
                        self.assertEqual(args[args.index('--model') + 1], 'claude-haiku-5-5')

    def test_python_paths_and_episode_record_keep_actual_response_models(self):
        spec = importlib.util.spec_from_file_location('claude_meta', ROOT / 'scripts/lib/claude_cli.py')
        meta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(meta)
        returned = {'claude-sonnet-5-5': {'inputTokens': 10}}
        calls, saved = [], []

        def run(args, **kwargs):
            calls.append(args)
            return SimpleNamespace(stdout=json.dumps({'structured_output': {'action': 'attack', 'paragraph': 0},
                'usage': {'input_tokens': 10, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
                          'output_tokens': 1}, 'modelUsage': returned}))

        env = {'json': json, 'subprocess': SimpleNamespace(run=run), 'model_id': meta.model_id,
               'model_metadata': meta.model_metadata, 'INSTRUCTIONS': 'Choose an action.', 'CRITERIA': {},
               'LLM_SCHEMA': '{}', 'CHILD_ENV': {}, 'ISOLATION_ARGS': [], 'DIR': Path('.'), 'EMPTY_CWD': '.', 'N_LLM': 1, 'N_QUESTIONS': 1,
               'write': lambda name, rows: saved.extend(rows), 'time': time,
               'ACTIONS': {'attack': [1]}, 'NOOP': [0], 'HOLD_TICS': 4, 'JEV_USD_PER_INPUT_TOKEN': .042/1e6,
               'observe': lambda game, state: {}, 'vzd': SimpleNamespace(GameVariable=SimpleNamespace(KILLCOUNT=0))}
        for file, names in [('recipe-doom.py', {'decide_llm', 'play'}), ('recipe-rerank.py', {'run_llm'})]:
            tree = ast.parse((ROOT / 'scripts' / file).read_text())
            selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
            exec(compile(ast.Module(body=selected, type_ignores=[]), file, 'exec'), env)
        action, info = env['decide_llm']({}, 'haiku')
        self.assertEqual(action, 'attack')
        self.assertEqual(info['models_used'], list(returned))
        env['run_llm']([{'id': 'q', 'article': 0, 'question': 'Which day?'}], [['Monday.']], 'haiku', n=1)
        self.assertEqual(saved[0]['requested_model'], 'claude-haiku-5-5')
        self.assertEqual(saved[0]['models_used'], list(returned))
        self.assertEqual(saved[0]['model_usage'], returned)
        for args in calls:
            self.assertEqual(args[args.index('--model')+1], 'claude-haiku-5-5')

        class Game:
            def new_episode(self): self.finished = False
            def is_episode_finished(self): return self.finished
            def get_state(self): return object()
            def set_action(self, action): pass
            def advance_action(self, count): self.finished = True
            def get_total_reward(self): return 0
            def get_game_variable(self, variable): return 0
            def is_player_dead(self): return False

        episode = env['play'](Game(), lambda obs: (action, info), 'llm:haiku')
        self.assertEqual(episode['models'], list(returned))
        self.assertEqual(episode['claude_calls'][0]['model_usage'], returned)


if __name__ == '__main__':
    unittest.main()
