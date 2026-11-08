#!/usr/bin/env python3
"""観測写真を改変せず並べた比較図と、構成図を出版用JPEGへ描画する。"""
import hashlib
import json
import argparse
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'experiments/recipes/so101'


def render(dest, *, diagram_only=False):
    dest.mkdir(parents=True, exist_ok=True)
    fonts = [p for p in Path('/System/Library/Fonts').glob('*.ttc')
             if 'ヒラギノ角ゴシック W3' in unicodedata.normalize('NFC', p.name)]
    if fonts:
        font = font_manager.FontProperties(fname=str(fonts[0]))
    else:
        font = font_manager.FontProperties(family='Noto Sans CJK JP')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    # 読者が手順ごとに「何の画像を・誰に渡し・何をするか」だけを読めるようにする
    boxes = [(.24, .79, .72, .16, '① 机を上から撮った画像 → Gemini\nキューブの4隅の位置を求める'),
             (.24, .56, .72, .15, '② 手元カメラの画像 → Jev\nキューブが写っているか（Noul）と、その位置（Box）'),
             (.24, .31, .72, .17, '③ 制御プログラム\n爪をキューブへ少し近づける\n掴める位置まで来たら、掴んで持ち上げる'),
             (.24, .06, .72, .16, '④ 持ち上げた後の手元カメラの画像 → Jev\nキューブを掴めているか（Choice）')]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                   linewidth=2, edgecolor='#243444', facecolor='#f4f7fa'))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontproperties=font, fontsize=17, linespacing=1.8)
    for start, end in [((.60,.775),(.60,.725)),
                       ((.60,.545),(.60,.495)),((.60,.295),(.60,.235))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=22,linewidth=2,color='#243444'))
    # ③から②へ戻る矢印。掴める位置に来るまで②と③を繰り返す
    ax.plot([.225,.175,.175],[.395,.395,.635],color='#243444',linewidth=2)
    ax.add_patch(FancyArrowPatch((.175,.635),(.225,.635),arrowstyle='-|>',
                                mutation_scale=22,linewidth=2,color='#243444'))
    ax.text(.085, .515, '掴める\n位置まで\n②③を\n繰り返す', ha='center', va='center',
            fontproperties=font, fontsize=15, linespacing=1.6)
    # 小さい注記は縮小すると読めないので、図に入れず本文と図の説明に書く
    fig.subplots_adjust(left=.025,right=.975,bottom=.035,top=.98)
    diagram = dest/'so101-flow.jpg'
    fig.savefig(diagram,format='jpg',dpi=160,pil_kwargs={'quality':95})
    plt.close(fig)
    with Image.open(diagram) as im:
        assert im.mode=='RGB' and max(im.size)>=1000 and im.width*im.height<4_000_000
    if diagram_only:
        print(diagram)
        return

    photos = [('repeat_fixed01_before.jpg','机上の赤いキューブ'),
              ('repeat_fixed01_verify_image.png','持ち上げた状態'),
              ('repeat_fixed01_verify_empty_image.png','戻した後：爪は空')]
    fig, axes = plt.subplots(1,3,figsize=(12,4.6),dpi=160)
    sources = []
    for axis, (name, title) in zip(axes,photos):
        path=ROOT/'images'/name
        axis.imshow(Image.open(path).convert('RGB'))
        axis.set_title(title,fontproperties=font,fontsize=16,pad=14)
        axis.axis('off')
        sources.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    fig.subplots_adjust(left=.008,right=.992,bottom=.07,top=.91,wspace=.018)
    figure = dest/'so101-grasp.jpg'
    fig.savefig(figure,format='jpg',dpi=160,pil_kwargs={'quality':95})
    plt.close(fig)
    metadata = {'selection':'First fixed-action trial, selected by trial order, not confidence or residual.',
                'sources':sources,'changes':'Publication layout and labels; entire original frames, no content reconstruction.',
                'photographs':'Captured in the author\'s SO-101 experiment environment on 2026-10-21.',
                'diagram_scope':'Fixed target hint for repeated trials; existing controller performs geometry and motion.'}
    (dest/'so101-grasp.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    for path in [diagram,figure]:
        with Image.open(path) as im:
            assert im.mode=='RGB' and max(im.size)>=1000 and im.width*im.height<4_000_000
        print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dest', type=Path)
    parser.add_argument('--diagram-only', action='store_true', help='写真の比較図は更新しない')
    args = parser.parse_args()
    render(args.dest, diagram_only=args.diagram_only)
