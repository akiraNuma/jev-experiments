"""比較の対象選択と、掲載コードと同じ感情ラベルの取り出し方を検証する。"""
import importlib.util
from pathlib import Path
import unittest
import json
import subprocess
import sys
import tempfile

spec=importlib.util.spec_from_file_location('report',Path(__file__).with_name('report-decisions-book.py'))
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class PublicationMetricsTest(unittest.TestCase):
    def test_common_ids_exclude_failure_for_every_method_not_incorrect_predictions(self):
        a={'a':{'result':{'ok':True,'predicted':'wrong'}},'b':{'result':{'ok':True}},'c':{'result':{'ok':True}}}
        b={'a':{'result':{'ok':True}},'b':{'result':{'ok':False}}}
        self.assertEqual(report.common({'a':a,'b':b},a),['a'])

    def test_refusal_is_kept_as_incorrect_and_held(self):
        a={'x':{'id':'x','label':'g','result':{'ok':True,'predicted':'g','confidence':.95}},
           'y':{'id':'y','label':'g','result':{'ok':True,'predicted':'g','confidence':.95}}}
        b={'x':{'id':'x','label':'g','result':{'ok':False,'error':'unexpected intent answer'}},
           'y':{'id':'y','label':'g','result':{'ok':True,'predicted':'g','confidence':.95}}}
        self.assertEqual(report.common({'a':a,'b':b},a),['x','y'])
        self.assertFalse(report.correct(b['x']))
        self.assertFalse(report.sure(b['x']))

    def test_refused_emotion_counts_as_not_labeled(self):
        answers={'joy':{'type':'predicate','probability':.9},'sad':{'type':'refusal'},'fear':{'type':'refusal'}}
        rows=[{'id':'a','labels':['joy','fear'],'result':{'ok':False,'error':'unexpected predicate answer',
               'refused_questions':['sad','fear'],'answers':answers}}]
        # 拒否した感情は付けない。正解の fear は見逃し、sad は誤検出にしない
        self.assertEqual(report.emotion_prediction(rows[0],['joy','sad','fear'],.5),{'joy'})
        result=report.emotion_metrics(rows,.5,['joy','sad','fear'])
        self.assertEqual(result['exact'],0)
        self.assertAlmostEqual(result['micro_f1'],2/3)
        # 回答が保存されていない拒否は、どの感情も付けなかったとみなす
        bare={'id':'b','labels':['joy'],'result':{'ok':False,'error':'unexpected predicate answer'}}
        self.assertEqual(report.emotion_prediction(bare,['joy','sad','fear'],.5),set())
        result=report.emotion_metrics([bare],.5,['joy','sad','fear'])
        self.assertEqual(result['exact'],0)
        self.assertEqual(result['micro_f1'],0)

    def test_emotion_fallback_and_extra_label_penalty(self):
        rows=[{'labels':['joy'],'result':{'scores':{'joy':.3,'sad':.2}}},
              {'labels':['sad'],'result':{'scores':{'joy':.7,'sad':.6}}}]
        result=report.emotion_metrics(rows,.5)
        self.assertEqual(result['exact'],1)
        self.assertAlmostEqual(result['accuracy'],.5)
        self.assertAlmostEqual(result['micro_f1'],.8)

    def test_toxicity_compares_only_ids_not_used_for_tuning(self):
        # しきい値は並び順で先頭の半分（奇数件なら少ない側）で選ぶので、比較は残りの後半だけ
        data_ids=['1','2','3','4','5']
        self.assertEqual(report.toxicity_heldout(['1','3','4','5'],data_ids),['3','4','5'])

    def test_paired_exact_test_known_values(self):
        self.assertEqual(report.exact_p([True]*3,[False]*3),.25)
        self.assertEqual(report.exact_p([True]*3,[True]*3),1)

    def test_cascade_entry_uses_the_same_common_ids_and_rejects_gold_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / 'experiments/classify'
            data.mkdir(parents=True)
            for dataset in ('banking77', 'massive-ja'):
                for mode in ('jev', 'decisions', 'llm-haiku', 'llm-sonnet', 'tfidf'):
                    rows = [{'id': id, 'label': 'gold', 'result': {'ok': not (id == 'failed' and mode == 'decisions'),
                             'predicted': 'wrong' if mode == 'jev' else 'gold', 'confidence': .2, 'cost_usd': .001}}
                            for id in ('valid', 'failed', 'missing') if not (id == 'missing' and mode == 'tfidf')]
                    (data / f'results-{dataset}-{mode}.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            cmd = [sys.executable, str(Path(__file__).with_name('recipe-cascade.py'))]
            result = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('banking77 jev: 1 件、単独正解 0/1、Haiku単独 1/1', result.stdout)
            self.assertIn('banking77 decisions: 1 件', result.stdout)
            self.assertIn('Haikuへ 1/1、正解 1/1', result.stdout)
            # 回答拒否は除かずHaikuへ回し、拒否した方式単独では不正解として数える
            target = data / 'results-banking77-decisions.jsonl'
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            rows[1]['result'] = {'ok': False, 'error': 'unexpected intent answer'}
            target.write_text('\n'.join(map(json.dumps, rows)))
            result = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('banking77 decisions: 2 件、単独正解 1/2', result.stdout)
            self.assertIn('Haikuへ 2/2、正解 2/2', result.stdout)
            target = data / 'results-banking77-decisions.jsonl'
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            rows[0]['label'] = 'different'
            target.write_text('\n'.join(map(json.dumps, rows)))
            result = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('正解ラベルが異なります', result.stderr)


if __name__=='__main__':unittest.main()
