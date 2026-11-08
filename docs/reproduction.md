# 測定に使ったスクリプト

測定をやり直すときや、本文の抜粋の前後を調べるときに使うファイルをまとめます。
スクリプトは、次の実験用リポジトリにあります。

<https://github.com/akiraNuma/jev-experiments/tree/v1>


次の表から、調べたい実験のファイルを選べます。
ファイル名は`scripts/`からの相対パスです。
「実行モード」は、測定や集計など、スクリプトに行わせる処理の名前です。

TypeScriptは`node --import tsx`で実行しました。
Pythonは`uv run`か、依存パッケージを入れた`python3`で実行しています。

## 本文の掲載値を保存結果から再計算する

保存結果からの再集計は、READMEの[保存済みの結果を読む・再集計する](../README.md#保存済みの結果を読む再集計する)にまとめています。
以下のスクリプト表は、新たに測定する場合や追加の比較を調べる場合に使えます。
新しい測定ではClaudeの指定IDと応答のモデル情報を保存します。
過去のClaudeの記録には応答した版がなく、新しい測定でその版を復元はできません。


<a id="scripts"></a>

実験とスクリプトの対応

| 実験 | スクリプト | 実行モード |
| --- | --- | --- |
| 1 振り分け | classify-judge.ts | jev、llm |
| 2 感情のラベル（28問を1回で） | recipe-multilabel.ts | jev、scaling、llm |
| 3 有害判定 | recipe-toxicity.ts | jev、llm |
| 4 並べ替え（一括送信） | recipe-rerank-batch.py | plan、run、report |
| 4 並べ替え（旧実装と比較相手） | recipe-rerank.py | bm25、embed、jev、llm |
| 5 答えの裏取り | recipe-halueval.ts | length、jev、jev-other、llm |
| 6 契約書の条項 | recipe-contract.ts | jev、jev-separate、llm |
| 7 値の抜き出し | recipe-extract.ts | first、jev、llm、report |
| 8 確信度でLLMへ | report-decisions-book.py | 本文の掲載値。recipe-cascade.pyは追加しきい値（契約日はrecipe-extract.tsのreport） |
| 9 写真の分類 | recipe-images.ts | text-1.13、text-1.14、examples-1.14 |
| 10 写真の位置取得 | recipe-images.ts | box-1.14 |
| 11 ゲームの操作 | recipe-doom.py | random、rule、jev、jev-v2、llm（syncで同期） |


実験1のTF-IDF分類器は`scripts/classify-baseline.py`です。
埋め込みを使う分類器は`scripts/classify-embed.py`にあります。
最小入力でLLMの待ち時間を測るスクリプトは、`scripts/recipe-latency-floor.ts`です。
実験7の測定では、正規表現で拾った日付の候補のうち、先頭から40個までを渡しました（`scripts/recipe-extract.ts`の`MAX_CANDIDATES`）。

<a id="rerank-batch"></a>

## 実験4の一括送信を再計測する

本のコードは`batch-local`方式です。
質問を共通のstateへ入れ、段落を各Noulへ渡します。
同じ300問について、旧実装と2種類の一括送信を比べるスクリプトを用意しました。

保存済みの結果を集計するだけなら、APIキーも元データも不要です。
次のコマンドはAPIを呼びません。

```sh
uv run --python 3.12 --with typesafe-sdk==0.6.0 scripts/recipe-rerank-batch.py report --output experiments/recipes/squad/batch-20261029
```

測り直す場合は、先に[元データの取得先](../experiments/DATA.md)からSQuAD 2.0のdev-v2.0.jsonを取得します。
`experiments/recipes/squad/data/dev-v2.0.json`へ置きます。
APIキーは`TYPESAFE_API_KEY`環境変数へ設定します。
このスクリプトは`.env`を読みません。

新しい出力先を指定して、先に条件だけを確認します。
`plan`はAPIを呼ばず、`run`がAPIを呼びます。

```sh
uv run --python 3.12 --with typesafe-sdk==0.6.0 scripts/recipe-rerank-batch.py plan --output experiments/recipes/squad/my-batch-run
uv run --python 3.12 --with typesafe-sdk==0.6.0 scripts/recipe-rerank-batch.py run --output experiments/recipes/squad/my-batch-run --budget-usd 1.0
```

費用・予算の指定・再実行の規則は、[再計測の記録](../experiments/recipes/squad/batch-20261029/README.md#再実行するときの規則)にあります。

## 画像を使う実験の実行準備

画像の実験を試すには、次の環境を用意します。

- Node.js 22系の22.12以降
- Python 3.11以降
- curl
- Pythonの実行と依存パッケージの導入に使うuv

uvの導入方法は、公式ガイド（<https://docs.astral.sh/uv/getting-started/installation/>）にあります。

取得・`npm ci`・APIキーの設定は、READMEの[最初のAPIリクエストを試す](../README.md#最初のapiリクエストを試す)と同じです。

写真の分類と位置の取得をまとめて測る場合は、次のコマンドを順に実行します。
写真の準備では、約811MBのデータをダウンロードします。
`--with pillow`は、画像処理に使うPillowを用意する指定です。

写真を準備した後の4つのコマンドが、Jevへリクエストを送ります。
最後のコマンドは、保存した判定結果を集計します。

```sh
uv run --with pillow python scripts/prepare-pets.py
node --import tsx scripts/recipe-images.ts text-1.13 eval
node --import tsx scripts/recipe-images.ts text-1.14 eval
node --import tsx scripts/recipe-images.ts examples-1.14 eval
node --import tsx scripts/recipe-images.ts box-1.14 eval
python3 scripts/recipe-images-report.py
```

選んだ写真のIDと正解枠は、`experiments/recipes/pets/manifest.json`に保存されます。
集計結果は、そのディレクトリの`report.json`で確認できます。

書籍の「写真の判定とロボットアーム」の該当節の実装では、このスクリプトから画像の送信部分を抜き出しています。
スクリプトには、動作確認用の`dev`と、画像用ヘッダーを外して送る`control`もあります。
レスポンスに返ったモデル名と座標も検証しています。
図を作り直すときは、`scripts/render-image-results.py`を使います。

## 文章の実験でClaudeを呼ぶ条件

Claudeを呼んだ条件は、[測定条件](measurement-method.md#compare)にあります。
