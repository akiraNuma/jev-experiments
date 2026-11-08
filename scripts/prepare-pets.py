#!/usr/bin/env python3
"""Oxford-IIIT Pet の固定部分集合を準備する。依存: Pillow。

python3 scripts/prepare-pets.py
画像・ライセンス・元アーカイブは git 対象外の data/ に置く。
評価の選定は API を呼ぶ前に固定し、モデルの答えでは選び直さない。
"""
import hashlib
import io
import json
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/recipes/pets"
DATA = OUT / "data"
BREEDS = {
    "British_Shorthair": "cat", "Russian_Blue": "cat", "Siamese": "cat",
    "beagle": "dog", "boxer": "dog", "pug": "dog",
}
BASE = "https://thor.robots.ox.ac.uk/pets/"
SOURCE = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
SEED = "jev-images-2026-10-15-v1"
ARCHIVE_SHA256 = {
    "annotations.tar.gz": "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91",
    "images.tar.gz": "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path):
    if path.exists():
        return
    print(f"download: {url}", flush=True)
    partial = path.with_suffix(path.suffix + ".part")
    # CC の配布サーバーはこの環境の urllib を403にする。curl で取得し、成功時だけ確定する。
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                    "--connect-timeout", "30", "--max-time", "600", url, "--output", str(partial)], check=True)
    partial.replace(path)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    for filename in ("annotations.tar.gz", "images.tar.gz"):
        download(BASE + filename, DATA / filename)
        if sha256(DATA / filename) != ARCHIVE_SHA256[filename]:
            raise ValueError(f"archive differs from the measured version: {filename}")
        # 完全なアーカイブか検査する。途中までの取得は成功として扱わない。
        with tarfile.open(DATA / filename, "r:gz") as archive:
            members = archive.getmembers()
        print(f"{filename}: {len(members)} entries", flush=True)
    download("https://creativecommons.org/licenses/by-sa/4.0/legalcode.en", DATA / "LICENSE.html")
    download(SOURCE, DATA / "source.html")

    with tarfile.open(DATA / "annotations.tar.gz", "r:gz") as archive:
        trainval = archive.extractfile("annotations/trainval.txt").read().decode()
        available = set(archive.getnames())
        selected = []
        for breed, species in BREEDS.items():
            ids = [line.split()[0] for line in trainval.splitlines()
                   if line and line.split()[0].rsplit("_", 1)[0] == breed
                   and f"annotations/xmls/{line.split()[0]}.xml" in available]
            ids.sort(key=lambda name: hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest())
            if len(ids) < 12:
                raise ValueError(f"not enough annotations: {breed}")
            for i, name in enumerate(ids[:12]):
                xml = ET.fromstring(archive.extractfile(f"annotations/xmls/{name}.xml").read())
                objects = xml.findall("object")
                if len(objects) != 1:
                    raise ValueError(f"expected one annotated head: {name}")
                bb = objects[0].find("bndbox")
                # VOC の1始まり・両端込みを0始まり・右下を含まない座標へ。
                box = [int(bb.findtext("xmin")) - 1, int(bb.findtext("ymin")) - 1,
                       int(bb.findtext("xmax")), int(bb.findtext("ymax"))]
                selected.append({"id": name, "breed": breed, "species": species,
                                 "split": "reference" if i == 0 else "dev" if i == 1 else "eval",
                                 "annotation_size": [int(xml.findtext("size/width")), int(xml.findtext("size/height"))],
                                 "source_box": box})

    by_name = {row["id"]: row for row in selected}
    with tarfile.open(DATA / "images.tar.gz", "r:gz") as archive:
        for member in archive:
            name = Path(member.name).stem
            if not member.isfile() or name not in by_name:
                continue
            raw = archive.extractfile(member).read()
            row = by_name[name]
            with Image.open(io.BytesIO(raw)) as image:
                image = image.convert("RGB")
                row["source_size"] = list(image.size)
                if row["source_size"] != row["annotation_size"]:
                    raise ValueError(f"annotation/image dimensions differ: {name}")
                row["sha256"] = hashlib.sha256(raw).hexdigest()
                for side in (384, 784):
                    # XML は元の画素座標。EXIF で回転せず、全体を等倍比率で縮放・白余白で正方形にする。
                    scale = side / max(image.size)
                    width, height = [round(v * scale) for v in image.size]
                    scaled = image.resize((width, height), Image.Resampling.LANCZOS)
                    left, top = (side - width) // 2, (side - height) // 2
                    canvas = Image.new("RGB", (side, side), "white")
                    canvas.paste(scaled, (left, top))
                    dest = DATA / f"{name}-{side}.jpg"
                    canvas.save(dest, "JPEG", quality=90)
                    row[f"sha256_{side}"] = hashlib.sha256(dest.read_bytes()).hexdigest()
                    x1, y1, x2, y2 = row["source_box"]
                    row[f"box_{side}"] = {
                        "left": (x1 * width / image.width + left) / side,
                        "top": (y1 * height / image.height + top) / side,
                        "right": (x2 * width / image.width + left) / side,
                        "bottom": (y2 * height / image.height + top) / side,
                    }
    if any("sha256" not in row for row in selected):
        raise ValueError("some selected images are missing")
    manifest = {"source": SOURCE, "license": "CC BY-SA 4.0", "seed": SEED,
                "selection": "per breed: SHA256(seed:id) ascending; 1 reference, 1 dev, 10 eval from trainval with XML",
                "preprocessing": "RGB JPEG quality 90, fit to square with white padding; 384 classification/reference, 784 Box",
                "archives": {name: sha256(DATA / name)
                             for name in ("annotations.tar.gz", "images.tar.gz")},
                "items": selected}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {len(selected)} images (6 reference, 6 dev, 60 eval)")


if __name__ == "__main__":
    prepare()
