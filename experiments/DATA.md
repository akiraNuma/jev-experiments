# 実験に使ったデータと利用条件

公開データセットの元データはリポジトリに入れていない（`.gitignore`）。測り直すときは、下の取得元から落として、表の場所に置く。SO-101実機編の自前の写真と記録は、末尾の節に記載したとおり含めている。

測定記録（`results-*.jsonl`など）には、採点に必要な範囲で元データの一部が入っている。
各データのIDと正解ラベルのほか、SQuAD 2.0の質問文（`experiments/recipes/squad/`）と、Oxford-IIIT Petの正解の枠（`pets/manifest.json`）を含む。
これらは元データのライセンスに従う。SQuAD 2.0とOxford-IIIT PetはCC BY-SA 4.0。
`experiments/` 以下のファイルはリポジトリのMIT Licenseの対象外で、それぞれの元データの条件に従う。
正解ラベルを持つ公開データを使い、ライセンスは取得元の表記に従う。

HaluEval の QA データでは、資料、質問、正しい答えは HotpotQA から取得されている。
誤った答えは LLM が生成したもので、すべての回答を人が作成・注釈したデータではない。
生成方法は [HaluEval の公式 README](https://github.com/RUCAIBox/HaluEval) の Overview と Data Release を参照。

| データ | ライセンス | 置く場所 | 使うスクリプト |
| --- | --- | --- | --- |
| Banking77（PolyAI） | CC BY 4.0 | `experiments/classify/data/banking77/`（`train.csv`、`test.csv`、`categories.json`） | `scripts/classify-*.ts`、`scripts/classify-*.py` |
| MASSIVE（Amazon）の ja-JP | CC BY 4.0 | `experiments/classify/data/massive/ja-JP.jsonl` | 同上 |
| GoEmotions（Google） | Apache 2.0 | `experiments/recipes/goemotions/data/`（`dev.tsv`、`test.tsv`、`emotions.txt`） | `scripts/recipe-multilabel.ts` |
| ContractNLI（Hitachi America） | CC BY 4.0 | `experiments/recipes/contractnli/data/test.json` | `scripts/recipe-contract.ts` |
| SQuAD 2.0（Stanford） | CC BY-SA 4.0 | `experiments/recipes/squad/data/dev-v2.0.json` | `scripts/recipe-rerank.py`、`scripts/recipe-rerank-batch.py` |
| CUAD v1（The Atticus Project） | CC BY 4.0 | `experiments/recipes/cuad/data/agreement-date-test.json` | `scripts/recipe-extract.ts` |
| LLM-jp Toxicity Dataset v2（LLM-jp） | CC BY 4.0 | `experiments/recipes/toxicity-ja/data/toxicity_dataset.jsonl` | `scripts/recipe-toxicity.ts` |
| HaluEval（RUCAIBox）の QA | MIT（元の HotpotQA は CC BY-SA 4.0） | `experiments/recipes/halueval/data/qa_data.json` | `scripts/recipe-halueval.ts` |
| Oxford-IIIT Pet（Parkhi et al.） | CC BY-SA 4.0 | `experiments/recipes/pets/data/` | `scripts/prepare-pets.py`、`scripts/recipe-images.ts` |

取得元:

- Banking77: https://github.com/PolyAI-LDN/task-specific-datasets
- MASSIVE: https://github.com/alexa/massive
- GoEmotions: https://github.com/google-research/google-research/tree/master/goemotions
- ContractNLI: https://stanfordnlp.github.io/contract-nli/
- SQuAD 2.0: https://rajpurkar.github.io/SQuAD-explorer/
- CUAD: https://www.atticusprojectai.org/cuad
- LLM-jp Toxicity Dataset v2: https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-toxicity-dataset-v2
- HaluEval: https://github.com/RUCAIBox/HaluEval
- Oxford-IIIT Pet: https://www.robots.ox.ac.uk/~vgg/data/pets/

LLM-jp の配布元は再配布を認めているが、日本国外のサーバーからの再配布に注意があるため、著者の判断でリポジトリに含めていない（配布元の README の License を参照）。

CUAD の `agreement-date-test.json` は、CUAD v1 の test.json（SQuAD の形式）から「Agreement Date」の問いだけを抜き出し、
契約ごとに `{ "title", "context", "answers": [{ "text", "answer_start" }] }` の配列にしたもの。中身は変えていない。

実験11（Doom）は ViZDoom（MIT、https://github.com/Farama-Foundation/ViZDoom）に同梱のシナリオを使うので、データは要らない。

画像の実験9・10は `uv run --with pillow python scripts/prepare-pets.py` で準備する。
公式の `images.tar.gz` と `annotations.tar.gz`、CC BY-SA 4.0のライセンス全文を `data/` に取得する。
6品種の見本6枚・動作確認6枚・評価60枚の選び方と加工方法は
`experiments/recipes/pets/README.md`、画像ID・ハッシュ・正解枠は同じ場所の `manifest.json` にある。
原写真の著作権は元の所有者に残る。画像の取得・加工と数値の再集計を分け、画像ファイルはgitに含めない。

## SO-101実機編の自前観測

`experiments/recipes/so101/`は著者の実験環境で取得した写真とAPIの記録で、公開データセット由来ではない。詳細は[so101のREADME](recipes/so101/README.md)にある。
