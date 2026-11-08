# 本の実験からコードを探す

番号と題名は『Jev実験レポート集』に合わせています。
[追加の比較表と採点内訳](detailed-results.md)、[共通の測定条件](measurement-method.md)、[再実行手順](reproduction.md)も参照できます。
過去の記録では実験の並びが違うため、当時の番号を本の番号として使わないでください。

## 実験と測定記録の対応

| 本の実験 | 測定するコード | 記録・根拠 |
| --- | --- | --- |
| 第1章の最初のリクエスト | [examples/first.ts](../examples/first.ts) | [basic-request](../experiments/recipes/basic-request/README.md)、F-044 |
| 1 問い合わせを分類する | [classify-judge.ts](../scripts/classify-judge.ts) | [classify](../experiments/classify/)、F-030 |
| 2 コメントに複数の感情ラベルを付ける | [recipe-multilabel.ts](../scripts/recipe-multilabel.ts) | [goemotions](../experiments/recipes/goemotions/)、F-031 |
| 3 日本語の文の有害さを判定する | [recipe-toxicity.ts](../scripts/recipe-toxicity.ts) | [toxicity-ja](../experiments/recipes/toxicity-ja/)、F-036 |
| 4 答えを含む段落を上位に並べる | [recipe-rerank-batch.py](../scripts/recipe-rerank-batch.py)、旧実装は[recipe-rerank.py](../scripts/recipe-rerank.py) | [一括送信の再計測](../experiments/recipes/squad/batch-20261029/)、旧記録は[squad](../experiments/recipes/squad/)、F-034 |
| 5 LLMの答えを資料で裏取りする | [recipe-halueval.ts](../scripts/recipe-halueval.ts) | [halueval](../experiments/recipes/halueval/)、F-037 |
| 6 契約書の条項を確認する | [recipe-contract.ts](../scripts/recipe-contract.ts) | [contractnli](../experiments/recipes/contractnli/)、F-032 |
| 7 文から値を抜き出す | [recipe-extract.ts](../scripts/recipe-extract.ts) | [cuad](../experiments/recipes/cuad/)、F-035 |
| 8 確信度が低いものをLLMに回す | [recipe-cascade.py](../scripts/recipe-cascade.py) | F-039。実験1・7で保存した結果を使う |
| 9 写真を分類する | [recipe-images.ts](../scripts/recipe-images.ts) | [pets](../experiments/recipes/pets/README.md)、F-040 |
| 10 写真から対象の位置を取り出す | [recipe-images.ts](../scripts/recipe-images.ts) | [pets](../experiments/recipes/pets/README.md)、F-041 |
| 11 ゲームの次の動作を選ぶ | [recipe-doom.py](../scripts/recipe-doom.py) | [doom](../experiments/recipes/doom/)、F-033 |
| SO-101で物を掴む | [report-so101.py](../scripts/report-so101.py) | [so101](../experiments/recipes/so101/README.md)、F-042・F-043 |

F番号の説明は[findings.md](../experiments/findings.md)にあります。
待ち時間の測り方はF-038、価格の補正条件は各F番号に残しています。

## 保存結果から再集計する

APIを呼ばない集計は、READMEの[保存済みの結果を読む・再集計する](../README.md#保存済みの結果を読む再集計する)から実行できます。

## 新しいAPIの判定を記録する

以下はAPIを呼ぶ測定です。先に[元データを準備](../experiments/DATA.md)します。
各スクリプトの冒頭に、使えるモードと出力先があります。

| 対象 | Jevを呼ぶ例 | 入力・注意 |
| --- | --- | --- |
| 問い合わせ分類 | `node --import tsx scripts/classify-judge.ts banking77 jev` | Banking77の評価用データ全件 |
| 感情ラベル | `node --import tsx scripts/recipe-multilabel.ts jev` | GoEmotions。28問をまとめる |
| 有害さ | `node --import tsx scripts/recipe-toxicity.ts jev` | LLM-jpの文章。取得元の条件を確認 |
| 段落選び | `uv run --python 3.12 --with typesafe-sdk==0.6.0 scripts/recipe-rerank-batch.py run --output experiments/recipes/squad/my-batch-run --methods batch-local` | SQuAD。1問分の段落の判定を1回で受け取る |
| 回答の裏取り | `node --import tsx scripts/recipe-halueval.ts jev` | HaluEval。回答1件の正誤判定 |
| 契約条項 | `node --import tsx scripts/recipe-contract.ts jev` | ContractNLI。17問をまとめる |
| 契約日 | `node --import tsx scripts/recipe-extract.ts jev` | CUADから作った契約日データ |

PythonのJev SDKを使う段落選びとゲームは、TypeScriptのSDKとは別の環境です。
その実行時に解決されたパッケージと版も、再測定の条件として記録してください。
Claudeを比べるモードでは、認証済みのClaude Code CLIも必要です。

画像の実験9・10には、[写真の準備と実行コマンド](../experiments/recipes/pets/README.md)があります。
画像の取得量は約811MBです。

実験11はViZDoomが必要です。
`recipe-doom.py`の冒頭に、実時間と同期の実行方法を記載しています。
本で測ったのは敵の位置などの数値から選ぶ操作で、ゲーム画面を画像として送る実験ではありません。

## SO-101の実機記録を確認する

このリポジトリにあるのは、著者の環境で保存した写真・判定・再集計コードです。
ロボットを動かす制御コードは含みません。
[so101のREADME](../experiments/recipes/so101/README.md)に、5回の固定試験と、別に試した失敗例の区別を説明しています。
