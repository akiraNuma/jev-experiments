"""実験11の図：ゲームの画面と、同じ時点でJevに渡す数値を並べる。

    uv run --python 3.12 --with vizdoom --with matplotlib scripts/render-doom-figure.py <出力先ディレクトリ>

画面は読者向けの図にだけ使う。実験ではJevに画像を渡していない。
数値は recipe-doom.py の observe() と同じ作り方で、同じ時点の画面から作る。
図用に大きい解像度で描くが、横位置と大きさは画面の幅・高さに対する割合なので、実験の320×240と同じ尺度になる。

場面の選び方：乱数の種を固定し、規則の方式（decide_rule）で同期モードのまま進め、
敵が2体以上見え、どの敵も画面の端で切れていない（横位置が-0.8〜0.8）最初の時点を使う。Jevの答えや見た目では選ばない。
"""

import hashlib
import json
import os
import random
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vizdoom as vzd
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

doom = import_module("recipe-doom")

SEED = 1
MIN_ENEMIES = 2
MAX_ABS_HORIZONTAL = 0.8


def capture():
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_1024X768)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_seed(SEED)
    game.set_available_buttons([vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK])
    game.init()
    random.seed(SEED)
    game.new_episode()
    tic = 0
    while not game.is_episode_finished():
        state = game.get_state()
        obs = doom.observe(game, state)
        enemies = obs["enemies_on_screen"]
        if len(enemies) >= MIN_ENEMIES and all(abs(e["horizontal"]) <= MAX_ABS_HORIZONTAL for e in enemies):
            screen = state.screen_buffer.copy()
            game.close()
            return screen, obs, tic
        action, _ = doom.decide_rule(obs)
        game.make_action(doom.ACTIONS[action], doom.HOLD_TICS)
        tic += doom.HOLD_TICS
    game.close()
    raise SystemExit("条件に合う時点がなかった")


def render(dest):
    dest.mkdir(parents=True, exist_ok=True)
    screen, obs, tic = capture()
    fonts = [p for p in Path("/System/Library/Fonts").glob("*.ttc")
             if "ヒラギノ角ゴシック W3" in unicodedata.normalize("NFC", p.name)]
    jp = font_manager.FontProperties(fname=str(fonts[0])) if fonts else font_manager.FontProperties(family="Noto Sans CJK JP")

    fig = plt.figure(figsize=(12, 5.4), dpi=160)
    ax_img = fig.add_axes([0.0, 0.04, 0.50, 0.84])
    ax_img.imshow(screen)
    ax_img.axis("off")
    ax_img.set_title("ゲームの画面", fontproperties=jp, fontsize=22, pad=12)

    ax_txt = fig.add_axes([0.53, 0.04, 0.47, 0.84])
    ax_txt.axis("off")
    ax_txt.set_title("同じ時点でJevに渡す数値", fontproperties=jp, fontsize=22, pad=12)
    text = json.dumps(obs, indent=2, ensure_ascii=False)
    ax_txt.text(0.02, 0.98, text, va="top", ha="left", family="Menlo", fontsize=17, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#f4f7fa", edgecolor="#243444"))

    out = dest / "doom-screen.jpg"
    fig.savefig(out, format="jpg", bbox_inches="tight", pad_inches=0.15, pil_kwargs={"quality": 92})
    plt.close(fig)

    meta = {
        "selection": f"seed={SEED}; rule policy in sync mode; first state with >= {MIN_ENEMIES} visible enemies, all with |horizontal| <= {MAX_ABS_HORIZONTAL} (tic {tic}). Not chosen by Jev output or appearance.",
        "scenario": "ViZDoom defend_the_center (ViZDoom code: MIT License)",
        "graphics": "Freedoom (freedoom2.wad bundled with ViZDoom), BSD-style license; Copyright (c) Contributors to the Freedoom project",
        "vizdoom_version": vzd.__version__,
        "observation": obs,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (dest / "doom-screen.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    render(Path(sys.argv[1]))
