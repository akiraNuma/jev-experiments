"""【計測用】振り分け（F-030）の Haiku の費用を、考える段込みと考える段なしで計算する。

    python3 scripts/classify-cost.py

`classify-judge.ts <データ> tokens:haiku -- --sample 100` の記録（トークン数つき）を使う。
- 問いの入力 = 記録の入力 - 前置き（空のディレクトリで測った値。recipes/goemotions/overhead-haiku.json）+ 構造化出力の道具の前置き 406
- 前置きは件ごとに数百トークン揺れるので、件ごとの差し引きでは毎回同じ部分を切り出せない。
  そこで、毎回変わる部分（問い合わせの文）を 0〜100 トークンの幅で置き、残りを毎回同じ部分（種類の一覧など）とする
- キャッシュあり: 毎回同じ部分をキャッシュ読みの単価、残りを入力の単価で数える（キャッシュの書き込みは件数で割ると小さいので除く）
- 考える段なし: 出力から考える段のトークンを引く。構造化出力の道具の呼び出しの分（約 60 トークン）は残る
"""

import json
import statistics as st
from pathlib import Path

IN, CACHE_READ, OUT = 0.10e-6, 0.01e-6, 0.50e-6  # Haiku 5.5、1 トークンあたりのドル
TOOL_PROMPT = 406
overhead = json.loads(Path("experiments/recipes/goemotions/overhead-haiku.json").read_text())["tokens"]

for name in ["banking77", "massive-ja"]:
    rows = [json.loads(l)["result"] for l in open(f"experiments/classify/results-{name}-tokens-haiku.jsonl")]
    p = st.mean(r["input_total"] - overhead + TOOL_PROMPT for r in rows)
    out_all = st.mean(r["output_tokens"] for r in rows)
    out_no_think = st.mean(r["output_tokens"] - r["thinking_tokens"] for r in rows)
    print(f"{name}: 問いの入力 平均 {p:.0f}、出力 平均 {out_all:.0f}（うち考える段 {out_all - out_no_think:.0f}）、API の時間 中央値 {st.median(r['duration_api_ms'] for r in rows):.0f} ms")
    for label, out in [("考える段なし", out_no_think), ("考える段込み", out_all)]:
        nocache = (p * IN + out * OUT) * 1000
        cached = [((p - d) * CACHE_READ + d * IN + out * OUT) * 1000 for d in (0, 100)]
        print(f"  {label}: 1,000 件あたり キャッシュなし ${nocache:.3f}、一覧をキャッシュ ${cached[0]:.3f}〜${cached[1]:.3f}")
