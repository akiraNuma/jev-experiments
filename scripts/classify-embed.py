"""【計測用】問い合わせの振り分けを、埋め込みとロジスティック回帰で測る。

    uv run --with scikit-learn --with sentence-transformers scripts/classify-embed.py

classify-baseline.py の TF-IDF より、少ない正解付きデータで伸びるかを見る。
埋め込みは intfloat/multilingual-e5-small（MIT、英日とも扱える小さいモデル）を手元の Mac で計算する。API は使わない。
学習の件数を減らすときは、classify-baseline.py と同じ種で抜く。
"""

import csv
import json
import random
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

DATA = Path("experiments/classify/data")
SIZES = [100, 300, 1000, 3000]
SEEDS = [0, 1, 2]


def banking77():
    def load(name):
        with open(DATA / "banking77" / name, newline="") as f:
            return [(r["text"], r["category"]) for r in csv.DictReader(f)]

    return load("train.csv"), load("test.csv")


def massive_ja():
    rows = [json.loads(l) for l in open(DATA / "massive" / "ja-JP.jsonl", encoding="utf-8") if l.strip()]
    return [(r["utt"], r["intent"]) for r in rows if r["partition"] == "train"], [(r["utt"], r["intent"]) for r in rows if r["partition"] == "test"]


model = SentenceTransformer("intfloat/multilingual-e5-small")
for name, load in [("banking77", banking77), ("massive-ja", massive_ja)]:
    train, test = load()
    started = time.time()
    # e5 は入力の頭に "query: " を付ける決まり
    e_train = model.encode([f"query: {t}" for t, _ in train], batch_size=64, normalize_embeddings=True)
    e_test = model.encode([f"query: {t}" for t, _ in test], batch_size=64, normalize_embeddings=True)
    print(f"{name}: 埋め込み {len(train) + len(test)} 件 {time.time() - started:.0f} 秒")
    index = {id(x): i for i, x in enumerate(train)}

    def score(subset):
        idx = [index[id(x)] for x in subset]
        clf = LogisticRegression(max_iter=3000, C=10).fit(e_train[idx], [train[i][1] for i in idx])
        return (clf.predict(e_test) == [l for _, l in test]).mean()

    for size in SIZES:
        scores = [score(random.Random(seed).sample(train, size)) for seed in SEEDS]
        print(f"  学習 {size} 件: 正解率 {sum(scores) / len(scores):.1%}（{', '.join(f'{s:.1%}' for s in scores)}）")
    print(f"  学習 全件 {len(train)} 件: 正解率 {score(train):.1%}")
