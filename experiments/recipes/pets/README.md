# 実験9・10：画像の分類と位置取得（2026-10-15）

要求: Jevの画像入力を実測し、本に分類と位置取得の2実験を追加する。
実験は既存のアプリと分離し、Anthropic APIは使わない。

## データと先に固定した条件

Oxford-IIIT Pet（Parkhi, Vedaldi, Zisserman, Jawahar, 2012）。
公式の配布条件はCC BY-SA 4.0。画像の著作権は元の所有者に残る。
出典: https://www.robots.ox.ac.uk/~vgg/data/pets/

- British Shorthair / Russian Blue / Siamese / beagle / boxer / pug の6品種。
- 公開された `trainval.txt` に含まれ、頭のXML注釈がある画像を対象とする。
  各品種内をSHA256(`jev-images-2026-10-15-v1:<id>`)の昇順に並べる。
  先頭1枚を見本、次の1枚を動作確認、次の10枚を評価とする。
- 合計72枚、評価60枚。公式test splitのベンチマークではなく、独自の小規模部分集合。
- 分類画像と見本は384四方、位置取得は784四方。縦横比を保ち、白余白を足す。
  JPEG RGB、品質90。元の画像サイズと頭の枠、変換後の枠をmanifestに残す。
- 正解の品種、ファイル名、ID、枠はstateに渡さない。
  送信画像名も `image-0.jpg` などの汎用名とする。
- 分類は6品種のChoice 1問。文章の選択肢を1.13と1.14で比較する。
  1.14では同じ説明に各品種1枚の見本を加える条件も測る。全条件で同じ60枚。
- 位置取得は1.14のみ。頭のBox、犬猫がいるかのNoul、自転車のBox、
  自転車があるかのNoulを1リクエストにまとめる。費用と時間は4問の合計。
  自転車の不在は評価写真の一覧を目視確認する。
- IoUは枠の共通部分の面積 / 和集合の面積。0.5以上の枚数も報告する。
  比較用の固定枠は左上(0.25, 0.25)、右下(0.75, 0.75)。評価前に固定した。
- HTTP multipartで直接呼ぶ。モデルを固定し、返されたモデルも検証する。
  同時実行2、再試行なし。各件を直ちに追記し、失敗も残す。
  同じ結果ファイルへの再実行は、失敗を含む既存IDを飛ばす。
- 結果を見て見本や問いを選び直さない。変更するなら新しい実験として分ける。
- LLMや専用の画像分類器との優劣は、この測定では主張しない。

## 再現

リポジトリルートで実行する。Node.js 22、Python/Pillow、curlを使う。
APIキーは通常どおり環境変数で設定する。画像・元アーカイブ・ライセンスは
git対象外の `data/` に置く。取得サイズは合計約811MB。

```sh
uv run --with pillow python scripts/prepare-pets.py
node --import tsx scripts/recipe-images.ts text-1.14 control
node --import tsx scripts/recipe-images.ts text-1.13 dev
node --import tsx scripts/recipe-images.ts text-1.14 dev
node --import tsx scripts/recipe-images.ts examples-1.14 dev
node --import tsx scripts/recipe-images.ts box-1.14 dev
# 動作確認後、下記を実行。正解率を使って問いを調整しない。
node --import tsx scripts/recipe-images.ts text-1.13 eval
node --import tsx scripts/recipe-images.ts text-1.14 eval
node --import tsx scripts/recipe-images.ts examples-1.14 eval
node --import tsx scripts/recipe-images.ts box-1.14 eval
python3 scripts/test-image-report.py
python3 scripts/recipe-images-report.py
```

`control` は画像用ヘッダーなしの1件。400エラーを確認するための負の対照。
結果には応答のモデル、全回答、usage、費用、送信・応答読込込みの時間を保存する。
画像のローカル読み込みとリクエスト組み立て、データ取得、前処理の時間は含めない。
APIの入力単価は1.13/1.14とも$0.042/Mtok、出力無料（2026-10-15確認）。
画像のトークン数はAPIのusageを使う。見本の分も費用に含む。

画像とBoxは実験機能であり、`TypeSafe-Danger-Zone-Images: true` が必要。
Boxにはnot-foundがないため、不在の対象への座標を有効な検出と扱わない。
Noulの0.5は今回の集計用であり、本番用に調整されたしきい値ではない。
同じ種の別個体が写る場合に、1個体の注釈とモデルの対象がずれる可能性がある。
事前学習データへの混入や、他の品種・複数対象・動画での精度は調べていない。

仕様: https://docs.typesafe.ai/images 、https://docs.typesafe.ai/primitives/box
