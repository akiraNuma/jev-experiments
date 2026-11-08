#!/usr/bin/env tsx
/**
 * 【計測用】claude -p の待ち時間の「床」を測る。
 *
 *   node --import tsx scripts/recipe-latency-floor.ts haiku sonnet
 *
 * レシピの LLM の待ち時間（duration_api_ms）には、claude -p が毎回足す前置き（約 2.6 万トークン、キャッシュから読む）の
 * 処理が入る。API を直接呼べば前置きは無い。どれだけ上乗せされているかの目安として、最小の入力と出力で 10 回呼び、
 * API の時間を記録する。
 *
 * 出力: experiments/recipes/latency/floor-<model>.jsonl
 */
import { writeFileSync } from "node:fs";
import { callClaude } from "./lib/claude-cli.js";

for (const model of process.argv.slice(2).filter((a) => a !== "--")) {
  const rows: unknown[] = [];
  for (let i = 0; i < 10; i++) {
    const r = await callClaude({ model, system: "Answer yes.", schema: { type: "object", properties: { a: { type: "boolean" } }, required: ["a"] }, input: "ok" });
    rows.push(r);
  }
  writeFileSync(`experiments/recipes/latency/floor-${model}.jsonl`, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  console.log(model, "done");
}
