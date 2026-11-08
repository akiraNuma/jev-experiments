#!/usr/bin/env tsx
/**
 * 【計測用】実験6：毎回違う長い文を読ませる。
 *
 *   node --import tsx scripts/recipe-contract.ts jev              # test 123 通、1 通に 17 問を 1 リクエストで
 *   node --import tsx scripts/recipe-contract.ts jev-separate     # 先頭 10 通だけ、1 問 1 リクエストで（まとめない場合の費用）
 *   node --import tsx scripts/recipe-contract.ts llm:haiku
 *   node --import tsx scripts/recipe-contract.ts overhead:haiku
 *
 * データ: ContractNLI（Hitachi America、CC BY 4.0）。秘密保持契約 123 通（test）それぞれに、17 個の決まった条項について
 * 「書いてある（Entailment）」「反対が書いてある（Contradiction）」「触れていない（NotMentioned）」を人が付けた。
 * 契約書は 1,231〜41,779 字、中央値 9,614 字。
 *
 * 方式:
 * - jev: 条項ごとに Choice を 1 問、17 問を契約書の state に投げる
 * - llm:<model>: 17 条項を system に並べ（毎回同じなのでキャッシュに乗る）、契約書を本文で渡し、17 個の答えを返させる
 *
 * 出力: experiments/recipes/contractnli/results-<方式>.jsonl。1 行が 1 通
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TypeSafeClient } from "@typesafe-ai/sdk";
import type { ChoiceQuestion } from "@typesafe-ai/sdk";
import { JEV_MODEL, jevCostUsd } from "../harness/jev.js";
import { callClaude, measureOverhead } from "./lib/claude-cli.js";
import { callDecisions, requireDecisionsKey } from "./lib/decisions-api.js";

const DIR = "experiments/recipes/contractnli";
const LABELS = ["Entailment", "Contradiction", "NotMentioned"] as const;

type Raw = {
  documents: { id: number; text: string; annotation_sets: { annotations: Record<string, { choice: string }> }[] }[];
  labels: Record<string, { hypothesis: string }>;
};
const raw = JSON.parse(readFileSync(join(DIR, "data/test.json"), "utf8")) as Raw;
const HYPOTHESES = Object.entries(raw.labels).map(([key, v]) => ({ key, text: v.hypothesis }));

type Doc = { id: string; text: string; gold: Record<string, string> };
const docs: Doc[] = raw.documents.map((d) => ({
  id: String(d.id),
  text: d.text,
  gold: Object.fromEntries(Object.entries(d.annotation_sets[0]?.annotations ?? {}).map(([k, a]) => [k, a.choice])),
}));

const CRITERIA = {
  Entailment: "The contract states this.",
  Contradiction: "The contract states the opposite of this.",
  NotMentioned: "The contract does not address this.",
};

function questionsFor(keys: string[]): Record<string, ChoiceQuestion> {
  return Object.fromEntries(
    HYPOTHESES.filter((h) => keys.includes(h.key)).map((h) => [
      h.key,
      { type: "choice", instructions: `Does this non-disclosure agreement say the following? "${h.text}"`, criteria: CRITERIA },
    ]),
  );
}

const client = new TypeSafeClient({ defaultModel: JEV_MODEL });

async function askJev(doc: Doc, keys: string[]) {
  const result = await client.systemOne({ state: { contract: doc.text }, questions: questionsFor(keys) });
  const answers = result.answers as Record<string, { choice: string; confidence: number }>;
  return { answers, model: result.model, input_tokens: result.usage.input_tokens };
}

async function judgeJev(doc: Doc): Promise<Record<string, unknown>> {
  const startedAt = Date.now();
  try {
    const { answers, model, input_tokens } = await askJev(doc, HYPOTHESES.map((h) => h.key));
    return {
      ok: true,
      model,
      predicted: Object.fromEntries(Object.entries(answers).map(([k, a]) => [k, a.choice])),
      confidence: Object.fromEntries(Object.entries(answers).map(([k, a]) => [k, a.confidence])),
      input_tokens,
      cost_usd: jevCostUsd(input_tokens),
      duration_ms: Date.now() - startedAt,
    };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error), duration_ms: Date.now() - startedAt };
  }
}

async function judgeJevSeparate(doc: Doc): Promise<Record<string, unknown>> {
  const startedAt = Date.now();
  try {
    const parts = await Promise.all(HYPOTHESES.map((h) => askJev(doc, [h.key])));
    const tokens = parts.reduce((a, p) => a + p.input_tokens, 0);
    return {
      ok: true,
      predicted: Object.fromEntries(parts.flatMap((p) => Object.entries(p.answers).map(([k, a]) => [k, a.choice]))),
      models: [...new Set(parts.map((p) => p.model))],
      input_tokens: tokens,
      cost_usd: jevCostUsd(tokens),
      duration_ms: Date.now() - startedAt,
    };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error), duration_ms: Date.now() - startedAt };
  }
}

const LLM_SYSTEM = [
  "You read a non-disclosure agreement. For each hypothesis below, answer Entailment if the contract states it,",
  "Contradiction if the contract states the opposite, or NotMentioned if the contract does not address it.",
  "",
  ...HYPOTHESES.map((h) => `${h.key}: ${h.text}`),
].join("\n");
const LLM_SCHEMA = {
  type: "object",
  properties: Object.fromEntries(HYPOTHESES.map((h) => [h.key, { type: "string", enum: LABELS }])),
  required: HYPOTHESES.map((h) => h.key),
};

async function judgeLlm(doc: Doc, model: string): Promise<Record<string, unknown>> {
  const r = await callClaude({ model, system: LLM_SYSTEM, schema: LLM_SCHEMA, input: doc.text });
  if (!r.ok) return r;
  const { output, ...rest } = r;
  return { ...rest, predicted: output };
}

async function judgeDecisions(doc: Doc): Promise<Record<string, unknown>> {
  const result = await callDecisions(JSON.stringify({ contract: doc.text }), HYPOTHESES.map((hypothesis) => ({
    type: "choice", name: hypothesis.key,
    instructions: `Does this non-disclosure agreement say the following? "${hypothesis.text}"`,
    choices: Object.entries(CRITERIA).map(([value, description]) => ({ value, description })),
  })));
  if (!result.ok) return result;
  if (HYPOTHESES.some((hypothesis) => result.answers[hypothesis.key]?.type !== "choice")) {
    return { ...result, ok: false, error: "unexpected contract answer" };
  }
  return { ...result, answers: undefined,
    predicted: Object.fromEntries(HYPOTHESES.map((hypothesis) => {
      const answer = result.answers[hypothesis.key];
      return [hypothesis.key, answer?.type === "choice" ? answer.choice : undefined];
    })),
    confidence: Object.fromEntries(HYPOTHESES.map((hypothesis) => {
      const answer = result.answers[hypothesis.key];
      return [hypothesis.key, answer?.type === "choice" ? answer.confidence : undefined];
    })) };
}

async function runAll(mode: string, targets: Doc[], judge: (d: Doc) => Promise<Record<string, unknown>>, concurrency: number) {
  const outPath = join(DIR, `results-${mode.replace(":", "-")}.jsonl`);
  const done = new Map<string, unknown>();
  if (existsSync(outPath)) {
    for (const line of readFileSync(outPath, "utf8").split("\n").filter((l) => l.trim() !== "")) {
      const row = JSON.parse(line) as { id: string; result: { ok?: boolean } };
      if (row.result.ok !== false) done.set(row.id, row);
    }
  }
  const queue = targets.filter((d) => !done.has(d.id));
  const rows: unknown[] = [];
  const startedAt = Date.now();
  await Promise.all(
    Array.from({ length: concurrency }, async () => {
      for (let doc = queue.shift(); doc !== undefined; doc = queue.shift()) {
        rows.push({ id: doc.id, chars: doc.text.length, gold: doc.gold, result: await judge(doc) });
      }
    }),
  );
  const all = [...done.values(), ...rows];
  writeFileSync(outPath, all.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const failed = rows.filter((r) => (r as { result: { ok?: boolean } }).result.ok === false).length;
  console.log(`${outPath}: ${all.length} 通（今回 ${rows.length} 通、失敗 ${failed} 通、${Math.round((Date.now() - startedAt) / 1000)} 秒）`);
}

const [mode = ""] = process.argv.slice(2).filter((a) => a !== "--");
if (mode === "jev") await runAll(mode, docs, judgeJev, 4);
else if (mode === "decisions") { requireDecisionsKey(); await runAll(mode, docs, judgeDecisions, 3); }
else if (mode === "jev-separate") await runAll(mode, docs.slice(0, 10), judgeJevSeparate, 1);
else if (mode.startsWith("llm:")) await runAll(mode, docs, (d) => judgeLlm(d, mode.slice(4)), 3);
else if (mode.startsWith("overhead:")) {
  const model = mode.slice(9);
  const measurement = await measureOverhead(model);
  writeFileSync(join(DIR, `overhead-${model}.json`), JSON.stringify({ model, ...measurement, measured_at: new Date().toISOString() }) + "\n");
  console.log(`${model} の前置き: ${measurement.tokens} トークン`);
} else throw new Error(`方式が分からない: ${mode}`);
