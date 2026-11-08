"""実験11：著者が同じ場面を手で遊び、倒した敵の数を記録する。

    uv run --python 3.12 --with vizdoom scripts/play-doom-human.py practice 3  # 練習3回（記録しない）
    uv run --python 3.12 --with vizdoom scripts/play-doom-human.py record     # 本番（10回まで記録）

ゲーム画面のウィンドウが開く。矢印キーの左右で向きを変え、Ctrlで撃つ。
マウスでの向き変えは切っている（ほかの方式と同じく、左右に向くか撃つかだけにするため）。
ゲームはリアルタイムで進む（一時停止の条件は人では作れないので測らない）。

人は画面を見て遊ぶ。ほかの方式には画面の画像を渡さず、数値だけを渡している。
人は練習で上達するので、この記録は検定に入れず参考として載せる。
本番の回数は遊ぶ前に10回と決めた。結果を見てから回数を変えない。

出力: experiments/recipes/doom/results-human.jsonl。1 行が 1 回
"""

import json
import os
import sys
import time
from pathlib import Path

import vizdoom as vzd

OUT = Path("experiments/recipes/doom")
RECORD_EPISODES = 10


def make_game():
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
    game.set_available_buttons([vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK])
    game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
    game.add_game_args("+use_mouse 0")
    game.init()
    return game


def play(game):
    game.new_episode()
    started = time.time()
    while not game.is_episode_finished():
        game.advance_action()
    return {
        "reward": game.get_total_reward(),
        "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
        "died": game.is_player_dead(),
        "seconds": round(time.time() - started, 1),
    }


def countdown(label):
    for n in (3, 2, 1):
        print(f"{label}：{n}秒後に始まります", flush=True)
        time.sleep(1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("practice", "record"):
        raise SystemExit("使い方: play-doom-human.py practice <回数> | record")
    record = sys.argv[1] == "record"
    practice = int(sys.argv[2]) if not record and len(sys.argv) > 2 else 1
    path = OUT / "results-human.jsonl"
    done = len(path.read_text().splitlines()) if path.exists() else 0
    if record and done >= RECORD_EPISODES:
        raise SystemExit(f"本番の{RECORD_EPISODES}回は記録済み: {path}")
    game = make_game()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if not record:
            for i in range(practice):
                countdown(f"練習 {i + 1}/{practice} 回目")
                result = play(game)
                print(f"練習: 倒した敵 {result['kills']}体（記録しません）")
            return
        with open(path, "a") as f:
            for i in range(done, RECORD_EPISODES):
                countdown(f"本番 {i + 1}/{RECORD_EPISODES} 回目")
                result = play(game)
                f.write(json.dumps({"mode": "human", "episode": i, **result}) + "\n")
                f.flush()
                print(f"本番 {i + 1}回目: 倒した敵 {result['kills']}体")
    finally:
        game.close()


if __name__ == "__main__":
    main()
