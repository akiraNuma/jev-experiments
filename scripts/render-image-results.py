#!/usr/bin/env python3
"""最初の評価画像を品種ごとに1枚ずつ示す。数値で図の対象を選び直さない。

python3 scripts/render-image-results.py <出力JPEGパス>
画像はCC BY-SA 4.0。加工内容と元画像IDを図内と隣のJSONに残す。
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DIR = Path(__file__).resolve().parents[1] / "experiments/recipes/pets"


def render(dest):
    manifest = json.loads((DIR / "manifest.json").read_text())
    report = json.loads((DIR / "report.json").read_text())
    rows = {r["id"]: r for r in map(json.loads, (DIR / "results-box-1.14-eval.jsonl").read_text().splitlines())}
    selected = {}
    for item in manifest["items"]:
        if item["split"] == "eval":
            selected.setdefault(item["breed"], item)
    canvas = Image.new("RGB", (1920, 1500), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=28)
    font = ImageFont.load_default(size=22)
    draw.text((24, 10), "Pet head: orange dashed = annotation; blue solid = Jev 1.14", fill="black", font=title_font)
    for n, item in enumerate(selected.values()):
        x, y = (n % 3) * 640, 58 + (n // 3) * 672
        image = Image.open(DIR / f'data/{item["id"]}-784.jpg').resize((624, 624), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + 8, y))
        gold = item["box_784"]
        predicted = rows[item["id"]]["result"]["answers"]["head"]["box"]
        for box, color, dashed in [(gold, "#d76b00", True), (predicted, "#0069d9", False)]:
            a, b, c, d = [round(v * 624) for v in (box["left"], box["top"], box["right"], box["bottom"])]
            a += x + 8; c += x + 8; b += y; d += y
            if dashed:
                for start in range(a, c, 18):
                    draw.line((start, b, min(start + 10, c), b), fill=color, width=5)
                    draw.line((start, d, min(start + 10, c), d), fill=color, width=5)
                for start in range(b, d, 18):
                    draw.line((a, start, a, min(start + 10, d)), fill=color, width=5)
                    draw.line((c, start, c, min(start + 10, d)), fill=color, width=5)
            else:
                draw.rectangle((a, b, c, d), outline=color, width=4)
        score = report["modes"]["box-1.14"]["iou_by_id"][item["id"]]
        draw.text((x + 12, y + 631), f'{item["id"]}   IoU {score:.2f}', fill="black", font=font)
    draw.text((24, 1410), "Oxford-IIIT Pet / Parkhi, Vedaldi, Zisserman, Jawahar (2012); photos: original owners.", fill="black", font=font)
    draw.text((24, 1440), "Resized, padded and annotated by Akira Numakura. Figure: CC BY-SA 4.0 (no warranty).", fill="black", font=font)
    draw.text((24, 1470), "Source: robots.ox.ac.uk/~vgg/data/pets/   License: creativecommons.org/licenses/by-sa/4.0/", fill="black", font=font)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, "JPEG", quality=95)
    dest.with_suffix(".json").write_text(json.dumps({
        "source": manifest["source"], "license": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Oxford-IIIT Pet / O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar (2012); photo copyright remains with original owners.",
        "changes": "Resized and padded source photographs; added annotation and Jev boxes, labels and layout. Adaptation: Akira Numakura, CC BY-SA 4.0. Provided without warranty.",
        "selection": "First evaluation image of each breed in the fixed manifest, independent of scores.",
        "source_ids": [item["id"] for item in selected.values()],
    }, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    render(Path(sys.argv[1]))
