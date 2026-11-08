#!/usr/bin/env tsx
/**
 * 【計測用】実験3：日本語の文の有害さを判定する（投稿の見張り、LLM の出力の見張り）。
 *
 *   node --import tsx scripts/recipe-toxicity.ts jev
 *   node --import tsx scripts/recipe-toxicity.ts llm:haiku -- --sample 300
 *   node --import tsx scripts/recipe-toxicity.ts overhead:haiku
 *
 * データ: LLM-jp Toxicity Dataset v2（LLM-jp、CC BY 4.0）。Common Crawl の日本語の文 3,847 件に、人が
 * 「有害（toxic）」「有害な表現を含むが全体としては有害でない（has_toxic_expression）」「有害でない（nontoxic）」と、
 * 有害さの種類 7 つ（猥褻、差別、暴力、違法、個人情報、企業の秘密、その他）を付けた。
 * 取得元: https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-toxicity-dataset-v2 の toxicity_dataset.jsonl。
 * 配布元の国外サーバーへの注意を踏まえ、著者の判断でラボの git には入れていない（experiments/DATA.md、.gitignore）。
 *
 * 文は 181〜50,958 字（中央値 1,680 字）。どの方式にも先頭 6,000 字だけを渡す。
 *
 * 方式:
 * - jev: 全体の判定を Choice 1 問、種類ごとに Noul 7 問。8 問を 1 リクエストで
 * - llm:<model>: 同じ定義を system に書き、全体の判定と 7 つの種類を返させる
 *
 * 出力: experiments/recipes/toxicity-ja/results-<方式>.jsonl
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TypeSafeClient } from "@typesafe-ai/sdk";
import type { Question } from "@typesafe-ai/sdk";
import { JEV_MODEL, jevCostUsd } from "../harness/jev.js";
import { callClaude, measureOverhead } from "./lib/claude-cli.js";
import { callDecisions, requireDecisionsKey } from "./lib/decisions-api.js";

const DIR = "experiments/recipes/toxicity-ja";
const CHAR_LIMIT = 6_000;
const LABELS = {
  toxic: "全体として有害な文。",
  has_toxic_expression: "有害になりうる表現を含むが、全体としては有害でない文。",
  nontoxic: "有害でない文。",
};
const CATEGORIES: Record<string, string> = {
  obscene: "露骨な性的表現や猥褻な内容があるか。",
  discriminatory: "差別的な表現や、他人への侮辱があるか。",
  violent: "暴力的な表現や脅しがあるか。",
  illegal: "違法、脱法、非倫理的な行為を勧める表現があるか。",
  personal: "個人情報やプライバシーをさらしているか。住所、メールアドレス、電話番号を含む。",
  corporate: "企業や団体の秘密の情報を漏らしているか。企業の住所、メールアドレス、電話番号を含む。",
  others: "上のどれにも当たらない、そのほかの有害な内容があるか。",
};

type Item = { id: string; text: string; label: string; categories: Record<string, boolean> };
const items: Item[] = readFileSync(join(DIR, "data/toxicity_dataset.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim() !== "")
  .map((l) => JSON.parse(l) as Record<string, string>)
  .map((r) => ({
    id: String(r.id),
    text: (r.text ?? "").slice(0, CHAR_LIMIT),
    label: r.label ?? "",
    categories: Object.fromEntries(Object.keys(CATEGORIES).map((k) => [k, r[k] === "yes"])),
  }));

const QUESTIONS: Record<string, Question> = {
  label: { type: "choice", instructions: "この日本語の文は、全体として有害か。", criteria: LABELS },
  ...Object.fromEntries(Object.entries(CATEGORIES).map(([k, v]) => [k, { type: "noul", instructions: `この日本語の文に、${v}` }])),
};

const client = new TypeSafeClient({ defaultModel: JEV_MODEL });

async function judgeJev(item: Item): Promise<Record<string, unknown>> {
  const startedAt = Date.now();
  try {
    const result = await client.systemOne({ state: { text: item.text }, questions: QUESTIONS });
    const a = result.answers as Record<string, { choice?: string; confidence?: number; probabilities?: Record<string, number>; noul?: number }>;
    return {
      ok: true,
      label: a.label?.choice,
      label_confidence: a.label?.confidence,
      label_probabilities: a.label?.probabilities,
      categories: Object.fromEntries(Object.keys(CATEGORIES).map((k) => [k, a[k]?.noul])),
      model: result.model, input_tokens: result.usage.input_tokens,
      cost_usd: jevCostUsd(result.usage.input_tokens),
      duration_ms: Date.now() - startedAt,
    };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error), duration_ms: Date.now() - startedAt };
  }
}

const LLM_SYSTEM = [
  "日本語の文が有害かを判定する。",
  "label は次のどれか 1 つ。",
  ...Object.entries(LABELS).map(([k, v]) => `- ${k}: ${v}`),
  "次の種類それぞれについて、当てはまれば true を返す。",
  ...Object.entries(CATEGORIES).map(([k, v]) => `- ${k}: ${v}`),
].join("\n");
const LLM_SCHEMA = {
  type: "object",
  properties: {
    label: { type: "string", enum: Object.keys(LABELS) },
    ...Object.fromEntries(Object.keys(CATEGORIES).map((k) => [k, { type: "boolean" }])),
  },
  required: ["label", ...Object.keys(CATEGORIES)],
};

async function judgeLlm(item: Item, model: string): Promise<Record<string, unknown>> {
  const r = await callClaude({ model, system: LLM_SYSTEM, schema: LLM_SCHEMA, input: item.text });
  if (!r.ok) return r;
  const { output, ...rest } = r;
  const o = output as Record<string, unknown>;
  return { ...rest, label: o.label, categories: Object.fromEntries(Object.keys(CATEGORIES).map((k) => [k, o[k] === true ? 1 : 0])) };
}

async function judgeDecisions(item: Item): Promise<Record<string, unknown>> {
  const result = await callDecisions(JSON.stringify({ text: item.text }), [
    { type: "choice", name: "label", instructions: "この日本語の文は、全体として有害か。",
      choices: Object.entries(LABELS).map(([value, description]) => ({ value, description })) },
    ...Object.entries(CATEGORIES).map(([name, description]) => ({
      type: "predicate" as const, name, instructions: `この日本語の文に、${description}`,
    })),
  ]);
  if (!result.ok) return result;
  const label = result.answers.label;
  if (label?.type !== "choice" || Object.keys(CATEGORIES).some((name) => result.answers[name]?.type !== "predicate")) {
    return { ...result, ok: false, error: "unexpected toxicity answer" };
  }
  return { ...result, answers: undefined, label: label.choice, label_confidence: label.confidence,
    label_probabilities: label.probabilities,
    categories: Object.fromEntries(Object.keys(CATEGORIES).map((name) => [name,
      result.answers[name]?.type === "predicate" ? result.answers[name].probability : undefined])) };
}

function sample(all: Item[], n: number): Item[] {
  const step = all.length / n;
  return Array.from({ length: n }, (_, k) => all[Math.floor(k * step)]).filter((i): i is Item => i !== undefined);
}

async function runAll(mode: string, targets: Item[], judge: (i: Item) => Promise<Record<string, unknown>>, concurrency: number) {
  const outPath = join(DIR, `results-${mode.replace(":", "-")}.jsonl`);
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
        // 正解だけを残す。文そのものは結果に書かない（有害な文を記録に増やさない）
        rows.push({ id: item.id, gold: { label: item.label, ...item.categories }, chars: item.text.length, result: await judge(item) });
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
// --first-half: データの並び順で前半（しきい値の調整に使う 1,923 件）。Jev と同じ手順で Decisions API のしきい値を選ぶため
const targets = sampleAt >= 0 ? sample(items, Number(rest[sampleAt + 1]))
  : rest.includes("--first-half") ? items.slice(0, Math.floor(items.length / 2)) : items;
if (mode === "jev") await runAll(mode, targets, judgeJev, 8);
else if (mode === "decisions") { requireDecisionsKey(); await runAll(mode, targets, judgeDecisions, 3); }
else if (mode.startsWith("llm:")) await runAll(mode, targets, (i) => judgeLlm(i, mode.slice(4)), 3);
else if (mode.startsWith("overhead:")) {
  const model = mode.slice(9);
  const measurement = await measureOverhead(model);
  writeFileSync(join(DIR, `overhead-${model}.json`), JSON.stringify({ model, ...measurement, measured_at: new Date().toISOString() }) + "\n");
  console.log(`${model} の前置き: ${measurement.tokens} トークン`);
} else throw new Error(`方式が分からない: ${mode}`);
