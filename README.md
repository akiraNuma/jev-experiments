# Jev実験レポート集 — サンプルコードと測定記録

書籍『Jev実験レポート集』の、11の実験とSO-101ロボットアームの観測記録です。
文章の分類、資料の照合、画像の判定などをJevで試し、正解率・費用・待ち時間を測りました。

まず1回APIを呼んでみることも、保存済みの結果をAPIなしで再集計することもできます。
本の本文原稿は含みません。

## 本に対応する版

本のリンクと以下の取得手順は、`v1`タグを使います。
`v1`は、本の手順と結果を変えない修正版へ進めます。各版の変更点は[CHANGELOG.md](CHANGELOG.md)にあります。

## 最初のAPIリクエストを試す

Node.js 22.12以降とnpmを用意します。
JevのAPIを呼ぶには、[TypeSafeのコンソール](https://console.typesafe.ai/keys)で発行したAPIキーが必要です。

```sh
git clone -b v1 https://github.com/akiraNuma/jev-experiments.git
cd jev-experiments
npm ci
export TYPESAFE_API_KEY="取得したAPIキー"
npm run example:first
```

detached HEADという注意が出ることがありますが、問題ありません。

[examples/first.ts](examples/first.ts)は、第1章の掲載コードです。
「キャッシュカードがまだ届きません」という文章を送り、種類と緊急性を1回のリクエストで尋ねます。
SDKは、本で使った`@typesafe-ai/sdk@0.6.0`に固定しています。
この例だけを試すなら、Claude Codeやロボットは不要です。

執筆時の1回の記録は431入力トークン、単価換算で約0.003円でした。現在の料金と無料枠はコンソールで確認してください。
APIキーはソースコード・実行ログ・スクリーンショットへ入れず、上の`export`が残るシェルの履歴も共有しないでください。

## 保存済みの結果を読む・再集計する

[実験ごとの案内](docs/experiments.md)から、本で読んだ実験を選べます。
追加の比較表と採点内訳は[詳しい測定結果](docs/detailed-results.md)にあります。
費用や待ち時間をどう数えたかは[測定条件](docs/measurement-method.md)で確認できます。
測定から読み取ったことは[experiments/findings.md](experiments/findings.md)にあります。
Decisions APIとの比較の集計条件は[Decisions APIとの比較](docs/decisions-results.md)にあります。

次の集計は、保存済みの回答を使います。JevやClaudeのAPIは呼びません。
Pythonの集計はPython 3.11以降の標準ライブラリで動きます。実験4だけは`uv`で依存を入れて動かします。
「データの準備が必要」と書いたものは、元データを[データの取得方法](experiments/DATA.md)に従って置いてから実行してください。
リポジトリのルートで実行してください。

```sh
# 第2章の掲載値（実験1・2・3・8）を、全方式で同じ評価IDにそろえて集計する（実験3のLLM-jpのデータの準備が必要）
python3 scripts/report-decisions-book.py

# 実験1：問い合わせ分類
node --import tsx scripts/classify-report.ts

# 実験2・6：感情ラベルと契約条項
python3 scripts/recipe-report.py

# 実験3：日本語の文の有害判定（LLM-jpのデータの準備が必要）
python3 scripts/recipe-toxicity-report.py

# 実験4：段落選び（本に掲載した一括送信の保存結果）
uv run --python 3.12 --with typesafe-sdk==0.6.0 scripts/recipe-rerank-batch.py report --output experiments/recipes/squad/batch-20261029

# 実験7：契約日（CUADのデータの準備が必要）
node --import tsx scripts/recipe-extract.ts report

# 実験8：問い合わせ分類の呼び分け
python3 scripts/recipe-cascade.py

# 実験9・10：写真の分類と頭の枠
python3 scripts/recipe-images-report.py

# 実験11：ゲーム
python3 scripts/recipe-doom-report.py

# SO-101：掴んだ状態を判定した記録
python3 scripts/report-so101.py
```

実験7の`report`は、元の契約書にある日付と照合して採点します。[CUADのデータ準備](experiments/DATA.md)を済ませてから実行してください。
画像とSO-101の集計は、同梱の記録と著者撮影の写真だけで動き、元画像・ロボット・カメラは使いません。

## データを用意して測定し直す

新たにモデルへ判定させる場合は、[実験ごとの案内](docs/experiments.md)と[データの取得方法](experiments/DATA.md)を使います。
画像の準備から測定までのコマンドは[再実行手順](docs/reproduction.md)にあります。
公開データセットの元データは同梱していません。

大量の判定を行うスクリプトは、APIを呼ぶコマンドか、保存結果を集計するコマンドかを区別してから実行してください。
保存済みの結果を再利用するスクリプトは全件を測り直すとは限らないため、別条件の測定では元の記録を保存し、出力先を分けます。
測定したモデルの版・日付・しきい値は本の数値を再現するための条件で、現在のモデルで同じ数値が出るという意味ではありません。
`findings.md`のF番号は測定当時の記録IDです。本の実験番号との対応は、[実験ごとの案内](docs/experiments.md)で示します。

## ライセンスとデータ

自作コードとその利用手順は[MIT License](LICENSE)です。
写真、APIの応答、測定記録、公開データ由来の資料は、MITの適用対象に含めません。
データの出典と利用条件は[experiments/DATA.md](experiments/DATA.md)にまとめています。
依存ライブラリのライセンスは、それぞれの配布元に従います。
