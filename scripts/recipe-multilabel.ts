#!/usr/bin/env tsx
/**
 * 【計測用】実験2：1 つの文に質問をたくさん投げる。
 *
 *   node --import tsx scripts/recipe-multilabel.ts jev            # test 全件、28 問を 1 リクエストで
 *   node --import tsx scripts/recipe-multilabel.ts scaling        # 問いの数を 1、7、14、28 と変えて、入力トークンと待ちを測る
 *   node --import tsx scripts/recipe-multilabel.ts llm:haiku -- --sample 300
 *   node --import tsx scripts/recipe-multilabel.ts overhead:haiku # claude -p の前置きを測る
 *   node --import tsx scripts/recipe-multilabel.ts jev -- --split dev  # 線を決めるための dev
 *
 * データ: GoEmotions（Google、Apache 2.0）。Reddit のコメントに、人が 28 種類の感情（neutral を含む）を付けた。
 * 1 件に複数の感情が付くことがある。test 5,427 件。
 *
 * 方式:
 * - jev: 感情ごとに Noul を 1 問、28 問を同じ state に投げる。0.5 以上を「その感情がある」とする
 * - llm:<model>: claude -p に感情の一覧を渡し、当てはまるものを配列で返させる
 *
 * 出力: experiments/recipes/goemotions/results-<方式>.jsonl
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TypeSafeClient } from "@typesafe-ai/sdk";
import type { NoulQuestion } from "@typesafe-ai/sdk";
import { JEV_MODEL, jevCostUsd } from "../harness/jev.js";
import { callClaude, measureOverhead } from "./lib/claude-cli.js";
import { callDecisions, requireDecisionsKey } from "./lib/decisions-api.js";

const DIR = "experiments/recipes/goemotions";
const EMOTIONS = readFileSync(join(DIR, "data/emotions.txt"), "utf8").split("\n").map((l) => l.trim()).filter(Boolean);

type Item = { id: string; text: string; labels: string[] };

// --split dev のときは dev を使う。0.5 の線が合わなかったので、線を dev で決めるため（test では決めない）
const SPLIT = process.argv.includes("dev") ? "dev" : "test";
const items: Item[] = readFileSync(join(DIR, `data/${SPLIT}.tsv`), "utf8")
  .split("\n")
  .filter((l) => l.trim() !== "")
  .map((l) => {
    const [text = "", ids = "", id = ""] = l.split("\t");
    return { id, text, labels: ids.split(",").map((i) => EMOTIONS[Number(i)] ?? "?") };
  });

function questionsFor(emotions: string[]): Record<string, NoulQuestion> {
  return Object.fromEntries(
    emotions.map((e) => [
      e,
      e === "neutral"
        ? { type: "noul", instructions: "Is this Reddit comment emotionally neutral, expressing none of the listed emotions?" }
        : { type: "noul", instructions: `Does this Reddit comment express ${e}?` },
    ]),
  );
}

const client = new TypeSafeClient({ defaultModel: JEV_MODEL });

async function judgeJev(item: Item, emotions: string[]): Promise<Record<string, unknown>> {
  const startedAt = Date.now();
  try {
    const result = await client.systemOne({ state: { comment: item.text }, questions: questionsFor(emotions) });
    const scores = Object.fromEntries(emotions.map((e) => [e, (result.answers as Record<string, { noul: number }>)[e]?.noul]));
    if (!Object.values(scores).every((v) => typeof v === "number")) return { ok: false, error: "応答の形が想定と違う" };
    return {
      ok: true,
      scores,
      predicted: emotions.filter((e) => (scores[e] ?? 0) >= 0.5),
      model: result.model, input_tokens: result.usage.input_tokens,
      cost_usd: jevCostUsd(result.usage.input_tokens),
      duration_ms: Date.now() - startedAt,
    };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error), duration_ms: Date.now() - startedAt };
  }
}

const LLM_SYSTEM = "You label the emotions expressed in a Reddit comment. Pick every emotion that applies. Use neutral only when none of the other emotions apply.";
const LLM_SCHEMA = {
  type: "object",
  properties: { emotions: { type: "array", items: { type: "string", enum: EMOTIONS }, minItems: 1 } },
  required: ["emotions"],
};

async function judgeLlm(item: Item, model: string): Promise<Record<string, unknown>> {
  const r = await callClaude({ model, system: LLM_SYSTEM, schema: LLM_SCHEMA, input: item.text });
  if (!r.ok) return r;
  const { output, ...rest } = r;
  return { ...rest, predicted: (output as { emotions?: string[] }).emotions ?? [] };
}

async function judgeDecisions(item: Item): Promise<Record<string, unknown>> {
  const result = await callDecisions(JSON.stringify({ comment: item.text }), EMOTIONS.map((name) => ({
    type: "predicate" as const, name,
    instructions: name === "neutral"
      ? "Is this Reddit comment emotionally neutral, expressing none of the listed emotions?"
      : `Does this Reddit comment express ${name}?`,
  })));
  if (!result.ok) return result;
  const scores = Object.fromEntries(EMOTIONS.map((name) => [name,
    result.answers[name]?.type === "predicate" ? result.answers[name].probability : undefined]));
  if (!Object.values(scores).every((value) => typeof value === "number")) {
    return { ...result, ok: false, error: "unexpected predicate answer" };
  }
  return { ...result, answers: undefined, scores,
    predicted: EMOTIONS.filter((name) => (scores[name] ?? 0) >= 0.5) };
}

function sample(all: Item[], n: number): Item[] {
  const step = all.length / n;
  return Array.from({ length: n }, (_, k) => all[Math.floor(k * step)]).filter((i): i is Item => i !== undefined);
}

async function runAll(mode: string, targets: Item[], judge: (i: Item) => Promise<Record<string, unknown>>, concurrency: number) {
  const outPath = join(DIR, `results-${mode.replace(":", "-")}${SPLIT === "dev" ? "-dev" : ""}.jsonl`);
  const done = new Map<string, unknown>();
  if (existsSync(outPath)) {
    for (const line of readFileSync(outPath, "utf8").split("\n").filter((l) => l.trim() !== "")) {
      const row = JSON.parse(line) as { id: string; result: { ok?: boolean } };
      if (row.result.ok !== false) done.set(row.id, row);
    }
  }
  const queue = targets.filter((i) => !done.has(i.id));
  const rows: unknown[] = [];
  const startedAt = Date.now();
  await Promise.all(
    Array.from({ length: concurrency }, async () => {
      for (let item = queue.shift(); item !== undefined; item = queue.shift()) {
        rows.push({ id: item.id, labels: item.labels, result: await judge(item) });
        if (rows.length % 500 === 0) console.log(`${rows.length}/${targets.length - done.size}`);
      }
    }),
  );
  const all = [...done.values(), ...rows];
  writeFileSync(outPath, all.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const failed = rows.filter((r) => (r as { result: { ok?: boolean } }).result.ok === false).length;
  console.log(`${outPath}: ${all.length} 件（今回 ${rows.length} 件、失敗 ${failed} 件、${Math.round((Date.now() - startedAt) / 1000)} 秒）`);
}

const [mode = "", ...rest] = process.argv.slice(2).filter((a) => a !== "--");
const sampleAt = rest.indexOf("--sample");
const targets = sampleAt >= 0 ? sample(items, Number(rest[sampleAt + 1])) : items;

if (mode === "jev") {
  await runAll(mode, targets, (i) => judgeJev(i, EMOTIONS), 8);
} else if (mode === "decisions") {
  requireDecisionsKey();
  await runAll(mode, targets, judgeDecisions, 3);
} else if (mode === "scaling") {
  // 問いの数だけを変える。感情の並びの先頭から取る（neutral は最後なので 28 のときだけ入る）
  const rows: unknown[] = [];
  for (const n of [1, 7, 14, 28]) {
    for (const item of sample(items, 50)) rows.push({ id: item.id, n, result: await judgeJev(item, EMOTIONS.slice(0, n)) });
  }
  writeFileSync(join(DIR, "results-scaling.jsonl"), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  console.log(`results-scaling.jsonl: ${rows.length} 件`);
} else if (mode.startsWith("llm:")) {
  await runAll(mode, targets, (i) => judgeLlm(i, mode.slice(4)), 3);
} else if (mode.startsWith("overhead:")) {
  const model = mode.slice(9);
  const measurement = await measureOverhead(model);
  writeFileSync(join(DIR, `overhead-${model}.json`), JSON.stringify({ model, ...measurement, measured_at: new Date().toISOString() }) + "\n");
  console.log(`${model} の前置き: ${measurement.tokens} トークン`);
} else {
  throw new Error(`方式が分からない: ${mode}`);
}
