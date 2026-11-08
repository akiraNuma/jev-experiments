"""【計測用】実験11：ゲームをリアルタイムに操作する。

    uv run --python 3.12 --with vizdoom --with typesafe-sdk==0.6.0 scripts/recipe-doom.py <方式> <回数> [sync]

方式: random / rule / jev / jev-v2 / decisions / decisions-own-v2 / llm:haiku
本の実験11の表は jev-v2（掲載コードの CRITERIA_V2）と decisions-own-v2、rule の結果。

ViZDoom（独自のコードは MIT、ZDoom 由来の部分は各配布元の条件）の defend_the_center を使う。プレイヤーは部屋の中央に立ち、左右に向きを変えるか撃つしかできない。
敵が四方から近づいてくる。倒すと +1、死ぬと -1。弾は 26 発、1 回は 2,100 tic（約 60 秒）で終わる。

ゲームは実時間（ASYNC_PLAYER、1 秒 35 tic）で進める。判定を待つあいだもゲームは止まらず、そのあいだは何もしない。
なので、判定が遅いほど敵に近づかれる。判定が届いたら、その動作を 4 tic（約 0.11 秒）続ける。

方式:
- random: 3 つの動作から無作為に選ぶ
- rule: 画面の中央近くに敵がいれば撃ち、いなければ一番大きく見える（近い）敵のほうへ向く。敵が見えなければ左を向く
- jev: 見えている敵の一覧（画面の横位置と大きさ）を state にして、動作を Choice で選ばせる
- jev-v2: jev と同じ入力で、撃つ範囲を数値で書いた選択肢の説明（本に掲載）を使う
- decisions: 同じ入力と jev-v2 の選択肢を OpenAI の Decisions API に渡す
- decisions-v1: 同じ入力と jev の 1 回目の選択肢（CRITERIA）を Decisions API に渡す
- decisions-own-v2: CRITERIA を、decisions-v1 の記録だけを見て 1 回直した選択肢（Jev の v2 と同じ手順）
- llm:<model>: 同じ一覧を claude -p に渡して選ばせる

Jev と LLM には、画面の画像ではなく、ViZDoom が返す「画面に映っている物の一覧」を文字で渡す。

出力: experiments/recipes/doom/results-<方式>.jsonl。1 行が 1 回
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import vizdoom as vzd
from typesafe_sdk import Choice, TypeSafeClient
from lib.decisions_api import create as create_decision, require_key as require_decisions_key
from lib.claude_cli import model_id, model_metadata

OUT = Path("experiments/recipes/doom")
ACTIONS = {"turn_left": [1, 0, 0], "turn_right": [0, 1, 0], "attack": [0, 0, 1]}
NOOP = [0, 0, 0]
HOLD_TICS = 4
JEV_USD_PER_INPUT_TOKEN = 0.042 / 1_000_000

INSTRUCTIONS = (
    "You are playing Doom. You stand still in the center of a round room and enemies walk toward you from all sides. "
    "You can only turn or fire. Which action should you take right now to kill the most enemies before they reach you?"
)
CRITERIA = {
    "attack": "Fire your pistol straight ahead. It only hits an enemy whose horizontal position is close to 0 (the center of the screen).",
    "turn_left": "Turn left, toward enemies with a negative horizontal position, or to look for enemies when none are visible.",
    "turn_right": "Turn right, toward enemies with a positive horizontal position.",
}

# 1 回目（jev）の記録で、敵が見えないときや画面の端の敵にも撃って弾を無駄にしていたので、撃つ条件を数字で書いた版
CRITERIA_V2 = {
    "attack": "Fire. Choose this only when some enemy has a horizontal position between -0.1 and 0.1. Otherwise the shot misses and wastes ammo.",
    "turn_left": "Turn left. Choose this when the largest (closest) enemy has a negative horizontal position, or when no enemy is visible.",
    "turn_right": "Turn right. Choose this when the largest (closest) enemy has a positive horizontal position.",
}


def make_game(sync=False):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_labels_buffer_enabled(True)
    # sync のときは判定を待つあいだゲームを止める。遅さの影響を除いて、判断の質だけを見るため
    game.set_mode(vzd.Mode.PLAYER if sync else vzd.Mode.ASYNC_PLAYER)
    game.set_available_buttons([vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK])
    game.init()
    return game


def observe(game, state):
    """画面に映っている敵を、横位置（-1 が左端、1 が右端）と大きさ（画面の高さに対する割合）で並べる"""
    width, height = game.get_screen_width(), game.get_screen_height()
    enemies = []
    for label in state.labels:
        if label.object_name in ("DoomPlayer",) or "Blood" in label.object_name or "Puff" in label.object_name:
            continue
        center = label.x + label.width / 2
        enemies.append(
            {
                "name": label.object_name,
                "horizontal": round((center - width / 2) / (width / 2), 2),
                "size": round(label.height / height, 2),
            }
        )
    enemies.sort(key=lambda e: -e["size"])
    return {
        "enemies_on_screen": enemies,
        "ammo": int(game.get_game_variable(vzd.GameVariable.AMMO2)),
        "health": int(game.get_game_variable(vzd.GameVariable.HEALTH)),
    }


# Decisions API 用の修正版。Jev の v2 と同じ手順で、CRITERIA を decisions-v1 の記録（10 回の trace）だけを見て 1 回直した。
# 記録では、中央から外れた敵や、敵がいないときに撃つ（48 回）、中央に敵がいるのに向きを変える（7 回）誤りが多く、
# 向く方向の誤りは 3 回だけだった。そこで撃つ条件だけを数値で書き、向く 2 つは CRITERIA のままにした。
DECISIONS_OWN_V2 = {
    "attack": "Fire. Choose this whenever any enemy has a horizontal position between -0.1 and 0.1. "
              "Do not fire when no enemy is visible or every enemy is farther from the center; the shot would miss.",
    "turn_left": CRITERIA["turn_left"],
    "turn_right": CRITERIA["turn_right"],
}


def decide_random(obs):
    return random.choice(list(ACTIONS)), {}


def decide_rule(obs):
    enemies = obs["enemies_on_screen"]
    if not enemies:
        return "turn_left", {}
    if any(abs(e["horizontal"]) < 0.1 for e in enemies) and obs["ammo"] > 0:
        return "attack", {}
    return ("turn_left" if enemies[0]["horizontal"] < 0 else "turn_right"), {}


# 省略すると SDK は jev-latest を使い、新しい版が出ると答えが変わる
JEV_MODEL = "jev-1.13.0"
jev_client = None


def decide_jev(obs, criteria=CRITERIA):
    global jev_client
    if jev_client is None:
        jev_client = TypeSafeClient()
    response = jev_client.system_one(
        model=JEV_MODEL, state=obs, questions={"action": Choice(instructions=INSTRUCTIONS, criteria=criteria)}
    )
    tokens = response.usage.input_tokens
    return response.choices["action"].choice, {"input_tokens": tokens, "model": response.model}


# 測定のエージェントと同じく、Jev のキーや Claude のセッション変数を子に渡さない
CHILD_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "TYPESAFE_"))}
# ユーザー階層の設定（~/.claude/CLAUDE.md など）を読ませない。scripts/lib/claude-cli.ts の ISOLATION_ARGS と同じ
ISOLATION_ARGS = ["--setting-sources", "project,local", "--strict-mcp-config"]
EMPTY_CWD = Path(tempfile.gettempdir()) / "jev-recipes-empty"
EMPTY_CWD.mkdir(exist_ok=True)
LLM_SCHEMA = json.dumps(
    {"type": "object", "properties": {"action": {"type": "string", "enum": list(ACTIONS)}}, "required": ["action"]}
)


def decide_llm(obs, model):
    system = INSTRUCTIONS + "\n" + "\n".join(f"{k}: {v}" for k, v in CRITERIA.items())
    requested_model = model_id(model)
    out = subprocess.run(
        ["claude", "-p", "--model", requested_model, *ISOLATION_ARGS, "--tools", "", "--system-prompt", system, "--json-schema", LLM_SCHEMA, "--output-format", "json"],
        input=json.dumps(obs),
        capture_output=True,
        text=True,
        env=CHILD_ENV,
        cwd=EMPTY_CWD,
    )
    parsed = json.loads(out.stdout)
    action = (parsed.get("structured_output") or {}).get("action", "turn_left")
    return action, {"duration_api_ms": parsed.get("duration_api_ms"),
                    **model_metadata(parsed, requested_model)}


def decide_decisions(obs, criteria=CRITERIA_V2):
    result = create_decision(json.dumps(obs), [{
        "type": "choice", "name": "action", "instructions": INSTRUCTIONS,
        "choices": [{"value": value, "description": description} for value, description in criteria.items()],
    }])
    answer = result["answers"]["action"]
    if answer.get("type") == "refusal":
        return "noop", {"model": result["model"], "input_tokens": result["input_tokens"],
                        "cost_usd": result["cost_usd"], "refused": True}
    if answer.get("type") != "choice" or answer.get("choice") not in ACTIONS:
        raise ValueError("unexpected action answer")
    return answer["choice"], {"model": result["model"], "input_tokens": result["input_tokens"],
                               "cost_usd": result["cost_usd"]}


def play(game, decide, mode):
    game.new_episode()
    decisions, latencies, tokens, trace, models = 0, [], 0, [], set()
    decisions_cost_usd = 0.0
    refusals = 0
    claude_calls = []
    started = time.time()
    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            game.advance_action(1)
            continue
        obs = observe(game, state)
        t0 = time.time()
        # 判定を待つあいだは何もしない。ゲームは実時間で進み続ける
        game.set_action(NOOP)
        action, info = decide(obs)
        latencies.append((time.time() - t0) * 1000)
        tokens += info.get("input_tokens", 0)
        decisions_cost_usd += info.get("cost_usd", 0)
        refusals += int(info.get("refused", False))
        if "model" in info:
            models.add(info["model"])
        if "requested_model" in info:
            models.update(info["models_used"])
            claude_calls.append({key: info[key] for key in ("requested_model", "models_used", "model_usage")})
        if len(trace) < 40:
            trace.append({"obs": obs, "action": action})
        decisions += 1
        if game.is_episode_finished():
            break
        game.set_action(ACTIONS.get(action, NOOP))
        game.advance_action(HOLD_TICS)
    seconds = time.time() - started
    latencies.sort()
    return {
        "reward": game.get_total_reward(),
        "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
        "died": game.is_player_dead(),
        "seconds": round(seconds, 1),
        "decisions": decisions,
        "refusals": refusals,
        "decisions_per_second": round(decisions / seconds, 2),
        "latency_ms_median": round(latencies[len(latencies) // 2]) if latencies else None,
        "jev_cost_usd": tokens * JEV_USD_PER_INPUT_TOKEN if mode.startswith("jev") else 0,
        "decisions_cost_usd": decisions_cost_usd,
        "models": sorted(models),
        "claude_calls": claude_calls,
        "trace": trace,
    }


def main():
    mode, episodes = sys.argv[1], int(sys.argv[2])
    if mode == "random":
        decide = decide_random
    elif mode == "rule":
        decide = decide_rule
    elif mode == "jev":
        decide = decide_jev
    elif mode == "jev-v2":
        decide = lambda obs: decide_jev(obs, CRITERIA_V2)  # noqa: E731
    elif mode.startswith("llm:"):
        decide = lambda obs: decide_llm(obs, mode[4:])  # noqa: E731
    elif mode == "decisions":
        require_decisions_key()
        decide = decide_decisions
    elif mode == "decisions-v1":
        require_decisions_key()
        decide = lambda obs: decide_decisions(obs, CRITERIA)  # noqa: E731
    elif mode == "decisions-own-v2":
        require_decisions_key()
        decide = lambda obs: decide_decisions(obs, DECISIONS_OWN_V2)  # noqa: E731
    else:
        raise SystemExit(f"方式が分からない: {mode}")
    OUT.mkdir(parents=True, exist_ok=True)
    sync = len(sys.argv) > 3 and sys.argv[3] == "sync"
    game = make_game(sync)
    path = OUT / f"results-{mode.replace(':', '-')}{'-sync' if sync else ''}.jsonl"
    completed = len(path.read_text().splitlines()) if mode.startswith("decisions") and path.exists() else 0
    with open(path, "a") as f:
        for i in range(completed, episodes):
            result = play(game, decide, mode)
            f.write(json.dumps({"mode": mode, "episode": i, **result}) + "\n")
            f.flush()
            print(json.dumps({k: v for k, v in result.items() if k != "trace"}))
    game.close()


if __name__ == "__main__":
    main()
