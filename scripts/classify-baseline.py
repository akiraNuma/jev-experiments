"""【計測用】問い合わせの振り分けを、学習データで作った古典的な分類器で測る。

    uv run --with scikit-learn scripts/classify-baseline.py

Jev や LLM は学習データを使わない。こちらは train で学習する。
「正解付きのデータが手元にあるなら、判定のモデルはそもそも要るか」を見るための比較。

- TF-IDF とロジスティック回帰。英語は単語の 1〜2 グラム、日本語は分かち書きをしないので文字の 1〜3 グラム
- 出力: experiments/classify/results-<データ>-tfidf.jsonl（classify-judge.ts と同じ形）
- 学習データを減らしたときの正解率も出す。何件の正解付きデータがあれば Jev を上回るかを見るため。
  減らすときは乱数の種を固定して 3 通り抜き、平均を出す
- C=10 を決めた経緯は残っていない。学習用データだけの交差検証で 10 と 100 の差が小さいことを
  2026-11-02 に確かめた（docs/detailed-results.md の「問い合わせの分類」）
"""

import csv
import json
import random
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DIR = Path("experiments/classify")
DATA = DIR / "data"


def banking77():
    def load(name):
        with open(DATA / "banking77" / name, newline="") as f:
            return [(r["text"], r["category"]) for r in csv.DictReader(f)]

    test = load("test.csv")
    return load("train.csv"), [(f"b{i}", t, l) for i, (t, l) in enumerate(test)], TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def massive_ja():
    rows = [json.loads(l) for l in open(DATA / "massive" / "ja-JP.jsonl", encoding="utf-8") if l.strip()]
    train = [(r["utt"], r["intent"]) for r in rows if r["partition"] == "train"]
    test = [(f"m{r['id']}", r["utt"], r["intent"]) for r in rows if r["partition"] == "test"]
    return train, test, TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)


SIZES = [100, 300, 1000, 3000]
SEEDS = [0, 1, 2]


def fit_predict(vectorizer, train, test):
    x_train = vectorizer.fit_transform([t for t, _ in train])
    model = LogisticRegression(max_iter=2000, C=10).fit(x_train, [l for _, l in train])
    return model.predict(vectorizer.transform([t for _, t, _ in test]))


for name, load in [("banking77", banking77), ("massive-ja", massive_ja)]:
    train, test, vectorizer = load()
    started = time.time()
    x_train = vectorizer.fit_transform([t for t, _ in train])
    model = LogisticRegression(max_iter=2000, C=10).fit(x_train, [l for _, l in train])
    train_s = time.time() - started

    started = time.time()
    predicted = model.predict(vectorizer.transform([t for _, t, _ in test]))
    predict_s = time.time() - started

    out = DIR / f"results-{name}-tfidf.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for (id_, _, label), p in zip(test, predicted):
            f.write(json.dumps({"id": id_, "label": label, "result": {"ok": True, "predicted": p}}, ensure_ascii=False) + "\n")
    accuracy = sum(p == l for (_, _, l), p in zip(test, predicted)) / len(test)
    print(f"{out}: 学習 {len(train)} 件 {train_s:.1f} 秒、判定 {len(test)} 件 {predict_s:.2f} 秒、正解率 {accuracy:.1%}")

    for size in SIZES:
        scores = []
        for seed in SEEDS:
            subset = random.Random(seed).sample(train, size)
            p = fit_predict(vectorizer, subset, test)
            scores.append(sum(a == l for (_, _, l), a in zip(test, p)) / len(test))
        print(f"  学習 {size} 件: 正解率 {sum(scores) / len(scores):.1%}（{', '.join(f'{s:.1%}' for s in scores)}）")
