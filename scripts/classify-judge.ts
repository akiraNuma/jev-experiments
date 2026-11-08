#!/usr/bin/env tsx
/**
 * 【計測用】エージェントの外での大量の判定として、問い合わせの振り分けを測る。
 *
 *   node --import tsx scripts/classify-judge.ts banking77 jev
 *   node --import tsx scripts/classify-judge.ts massive-ja jev
 *   node --import tsx scripts/classify-judge.ts banking77 llm:haiku -- --sample 400
 *
 * データ（どちらも CC BY 4.0、人が正解を付けたもの。experiments/classify/data/ に LICENSE ごと置いた）:
 * - banking77: PolyAI の銀行への問い合わせ。英語、77 種類。test 3,080 件
 * - massive-ja: Amazon の MASSIVE の ja-JP。音声アシスタントへの話しかけ。60 種類。test 2,974 件
 *
 * 方式:
 * - jev: 種類の名前を並べた Choice 1 問。1 件 1 リクエスト
 * - llm:<model>: claude -p に同じ種類の名前を渡し、JSON スキーマの enum で 1 つ選ばせる
 * - tokens:<model>: llm と同じ問いを scripts/lib/claude-cli.ts で投げ、考える段と出力のトークン数を記録する。
 *   llm の回はトークン数を記録していなかったので、考える段込みの費用を出すために後から足した（--sample 100 で使う）
 *
 * 両方とも、渡すのは種類の名前だけ。説明や例は付けない（どちらにも同じ材料を渡すため）。
 * 学習データは使わない。学習データを使う方式は scripts/classify-baseline.py で別に測る。
 *
 * 出力: experiments/classify/results-<データ>-<方式>.jsonl。途中で止まっても、済んだ分は使い回す。
 */
import { spawn } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TypeSafeClient } from "@typesafe-ai/sdk";
import type { ChoiceQuestion } from "@typesafe-ai/sdk";
import { JEV_MODEL, jevCostUsd } from "../harness/jev.js";
import { callClaude, claudeModelId, claudeModelsUsed, EMPTY_CWD, ISOLATION_ARGS } from "./lib/claude-cli.js";
import { callDecisions, requireDecisionsKey } from "./lib/decisions-api.js";

const DIR = "experiments/classify";
const DATA = join(DIR, "data");

type Item = { id: string; text: string; label: string };
type Dataset = { items: Item[]; labels: string[]; instructions: string; llmSystem: string };

/** 引用符と、その中のカンマ・改行に対応した最小の CSV 読み */
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (c !== "\r") field += c;
  }
  if (field !== "" || row.length > 0) rows.push([...row, field]);
  return rows;
}

function loadDataset(name: string): Dataset {
  if (name === "banking77") {
    const rows = parseCsv(readFileSync(join(DATA, "banking77/test.csv"), "utf8")).slice(1).filter((r) => r.length === 2);
    const labels = JSON.parse(readFileSync(join(DATA, "banking77/categories.json"), "utf8")) as string[];
    return {
      items: rows.map(([text, label], i) => ({ id: `b${i}`, text: text ?? "", label: label ?? "" })),
      labels,
      instructions: "This is a message a customer sent to an online bank. Which intent category does it belong to?",
      llmSystem: "You classify a message a customer sent to an online bank into exactly one intent category. Reply with the category only.",
    };
  }
  if (name === "massive-ja") {
    const rows = readFileSync(join(DATA, "massive/ja-JP.jsonl"), "utf8")
      .split("\n")
      .filter((l) => l.trim() !== "")
      .map((l) => JSON.parse(l) as { id: string; partition: string; intent: string; utt: string })
      .filter((r) => r.partition === "test");
    // 種類の一覧は train から取る。test にだけ無い種類があっても選択肢から消えないように
    const all = readFileSync(join(DATA, "massive/ja-JP.jsonl"), "utf8")
      .split("\n")
      .filter((l) => l.trim() !== "")
      .map((l) => (JSON.parse(l) as { intent: string }).intent);
    return {
      items: rows.map((r) => ({ id: `m${r.id}`, text: r.utt, label: r.intent })),
      labels: [...new Set(all)].sort(),
      instructions: "これは、ユーザーが音声アシスタントに話しかけた言葉である。どの意図に当たるか。",
      llmSystem: "ユーザーが音声アシスタントに話しかけた言葉を、意図の種類 1 つに振り分ける。種類だけを答える。",
    };
  }
  throw new Error(`データが分からない: ${name}`);
}

// ---- jev ----

const client = new TypeSafeClient({ defaultModel: JEV_MODEL });

async function judgeJev(item: Item, question: ChoiceQuestion): Promise<Record<string, unknown>> {
  const startedAt = Date.now();
  try {
    const result = await client.systemOne({ state: { message: item.text }, questions: { intent: question } });
    const answer = (result.answers as Record<string, { choice?: string; confidence?: number; probabilities?: Record<string, number> }>).intent;
    if (answer?.choice === undefined) return { ok: false, error: "応答の形が想定と違う", duration_ms: Date.now() - startedAt };
    const top3 = Object.entries(answer.probabilities ?? {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    return {
      ok: true,
      model: result.model,
      predicted: answer.choice,
      confidence: answer.confidence,
      top3,
      input_tokens: result.usage.input_tokens,
      cost_usd: jevCostUsd(result.usage.input_tokens),
      duration_ms: Date.now() - startedAt,
    };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error), duration_ms: Date.now() - startedAt };
  }
}

// ---- llm ----

// 測定のエージェントと同じく、Jev のキーや Claude のセッション変数を子に渡さない
const CHILD_ENV = Object.fromEntries(Object.entries(process.env).filter(([k]) => !/^(CLAUDE|TYPESAFE_)/.test(k)));

function judgeLlm(item: Item, model: string, dataset: Dataset): Promise<Record<string, unknown>> {
  const startedAt = Date.now();
  const requestedModel = claudeModelId(model);
  const schema = JSON.stringify({
    type: "object",
    properties: { intent: { type: "string", enum: dataset.labels } },
    required: ["intent"],
  });
  return new Promise((resolve) => {
    const child = spawn(
      "claude",
      ["-p", "--model", requestedModel, ...ISOLATION_ARGS, "--tools", "", "--system-prompt", dataset.llmSystem, "--json-schema", schema, "--output-format", "json"],
      { env: CHILD_ENV, cwd: EMPTY_CWD, stdio: ["pipe", "pipe", "pipe"] },
    );
    let out = "";
    child.stdout.on("data", (d: Buffer) => (out += d.toString()));
    child.stdin.end(item.text);
    child.on("close", () => {
      try {
        const parsed = JSON.parse(out) as { structured_output?: { intent?: string }; total_cost_usd?: number; is_error?: boolean; result?: string; modelUsage?: unknown };
        const predicted = parsed.structured_output?.intent;
        if (parsed.is_error === true || predicted === undefined) {
          resolve({ ok: false, error: String(parsed.result ?? "no intent").slice(0, 200), duration_ms: Date.now() - startedAt });
          return;
        }
        resolve({ ok: true, predicted, requested_model: requestedModel,
          models_used: claudeModelsUsed(parsed.modelUsage), model_usage: parsed.modelUsage ?? null,
          cost_usd: parsed.total_cost_usd, duration_ms: Date.now() - startedAt });
      } catch {
        resolve({ ok: false, error: out.slice(0, 200), duration_ms: Date.now() - startedAt });
      }
    });
  });
}

// ---- main ----

/** 等間隔で抜く。乱数を使わず、どの方式でも同じ件を選ぶ */
function sample(items: Item[], n: number): Item[] {
  const step = items.length / n;
  return Array.from({ length: n }, (_, k) => items[Math.floor(k * step)]).filter((i): i is Item => i !== undefined);
}

async function main() {
  const [datasetName = "", mode = "", ...rest] = process.argv.slice(2).filter((a) => a !== "--");
  const dataset = loadDataset(datasetName);
  const sampleAt = rest.indexOf("--sample");
  const items = sampleAt >= 0 ? sample(dataset.items, Number(rest[sampleAt + 1])) : dataset.items;
  const question: ChoiceQuestion = {
    type: "choice",
    instructions: dataset.instructions,
    criteria: Object.fromEntries(dataset.labels.map((l) => [l, null])),
  };
  if (mode === "decisions") requireDecisionsKey();

  const outPath = join(DIR, `results-${datasetName}-${mode.replace(":", "-")}.jsonl`);
  const done = new Map<string, unknown>();
  if (existsSync(outPath)) {
    for (const line of readFileSync(outPath, "utf8").split("\n").filter((l) => l.trim() !== "")) {
      const row = JSON.parse(line) as { id: string; result: { ok?: boolean } };
      if (row.result.ok !== false) done.set(row.id, row);
    }
  }

  const judge = (item: Item): Promise<Record<string, unknown>> => {
    if (mode === "jev") return judgeJev(item, question);
    if (mode === "decisions") {
      return callDecisions(JSON.stringify({ message: item.text }), [{
        type: "choice", name: "intent", instructions: dataset.instructions,
        choices: dataset.labels.map((value) => ({ value })),
      }]).then((r) => {
        if (!r.ok) return r;
        const answer = r.answers.intent;
        if (answer?.type !== "choice") return { ...r, ok: false, error: "unexpected intent answer" };
        return { ...r, answers: undefined, predicted: answer.choice, confidence: answer.confidence,
          top3: [...answer.probabilities].sort((a, b) => b.probability - a.probability)
            .slice(0, 3).map((entry) => [entry.value, entry.probability]) };
      });
    }
    if (mode.startsWith("llm:")) return judgeLlm(item, mode.slice(4), dataset);
    if (mode.startsWith("tokens:")) {
      const schema = { type: "object", properties: { intent: { type: "string", enum: dataset.labels } }, required: ["intent"] };
      return callClaude({ model: mode.slice(7), system: dataset.llmSystem, schema, input: item.text }).then((r) =>
        r.ok ? { ...r, output: undefined, predicted: (r.output as { intent?: string }).intent } : r,
      );
    }
    throw new Error(`方式が分からない: ${mode}`);
  };

  const rows: unknown[] = [];
  const queue = items.filter((i) => !done.has(i.id));
  const startedAt = Date.now();
  let finished = 0;
  const concurrency = mode === "decisions" || mode.startsWith("llm:") || mode.startsWith("tokens:") ? 3 : 8;
  await Promise.all(
    Array.from({ length: concurrency }, async () => {
      for (let item = queue.shift(); item !== undefined; item = queue.shift()) {
        const result = await judge(item);
        rows.push({ id: item.id, label: item.label, result });
        if (++finished % 200 === 0) console.log(`${finished}/${items.length - done.size}`);
      }
    }),
  );

  const all = [...done.values(), ...rows];
  writeFileSync(outPath, all.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const failed = rows.filter((r) => (r as { result: { ok?: boolean } }).result.ok === false).length;
  console.log(`${outPath}: ${all.length} 件（今回 ${rows.length} 件、失敗 ${failed} 件、並列 ${concurrency} で ${Math.round((Date.now() - startedAt) / 1000)} 秒）`);
}

await main();
