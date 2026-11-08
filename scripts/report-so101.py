#!/usr/bin/env python3
"""保存したSO-101観測を再集計する。API・カメラ・アームには接続しない。"""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'experiments/recipes/so101'


def stats(values):
    return {'n': len(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'min': min(values), 'max': max(values),
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


def main():
    data = json.loads((ROOT / 'observations.json').read_text())
    cases, trials = data['cases'], data['trials']
    if len(cases) != 10 or len(trials) != 5 or {t['trial'] for t in trials} != set(range(1, 6)):
        raise ValueError('Expected five trials and ten image cases')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    for case in cases:
        actual = hashlib.sha256((ROOT / case['image']).read_bytes()).hexdigest()
        if actual != case['image_sha256']:
            raise ValueError(f"Image hash mismatch: {case['id']}")
    result = {
        'source_sha256': hashlib.sha256((ROOT / 'observations.json').read_bytes()).hexdigest(),
        'physical_trials': len(trials), 'sensor_gripped': sum(t['sensor_gripped'] for t in trials),
        'image_cases': len(cases),
        'label_source': 'Unblinded Codex visual review, not independent human annotation.',
        'models': {},
        'pick_http_seconds': stats([t['http_seconds']['pick'] for t in trials]),
        'servo_seconds': stats([t['pick_record']['servo_s'] for t in trials]),
        'self_estimated_error_mm': stats([t['pick_record']['final_error_mm'] for t in trials]),
        'cycle_seconds': stats([t['cycle_seconds'] for t in trials]),
        'limits': data['limits'],
    }
    for model in ['jev', 'gemini']:
        values = [c[model] for c in cases]
        result['models'][model] = {
            'models': sorted({v['model'] for v in values}),
            'latency_seconds': stats([v['latency_s'] for v in values]),
            'estimated_verification_cost_usd': sum(v['cost_usd'] for v in values),
            'agreement_with_provisional_labels': sum(c[model]['verdict'] == c['provisional_visual_label'] for c in cases),
            'by_state': {
                label: {'n': sum(c['provisional_visual_label'] == label for c in cases),
                        'agreement': sum(c['provisional_visual_label'] == label and c[model]['verdict'] == label for c in cases)}
                for label in ['success', 'failure']
            },
        }
    path = ROOT / 'report.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
