#!/usr/bin/env tsx
/**
 * 【計測用】実験5：LLM の答えを、手元の資料で裏取りする（RAG の答えの見張り）。
 *
 *   node --import tsx scripts/recipe-halueval.ts length | jev | jev-other | llm:haiku -- --sample 300 | overhead:haiku | report
 *
 * jev-other は、同じ質問のもう片方の答え（偶数番目なら誤った答え、奇数番目なら正しい答え）を判定する。
 * jev と合わせると、同じ質問の 2 つの答えのどちらを「支えられている」と高く見たかで比べられる。
 * 長さの偏りの影響を、件ごとの正誤より受けにくい。
 *
 * データ: HaluEval（RUCAIBox、MIT）の QA。元は HotpotQA（CC BY-SA 4.0）。資料（knowledge）、質問、正しい答え、
 * もっともらしい誤った答えの組が 10,000 件ある。偶数番目は正しい答え、奇数番目は誤った答えを判定させる（半々になる）。
 *
 * 誤った答えは文になっていることが多く、正しい答えは短い語句が多い。長さだけで当てられてしまうので、
 * 「答えが 5 語より長ければ誤り」とだけ判定する length も並べる。
 *
 * 方式:
 * - length: 答えの語数だけで判定する
 * - jev: 「資料が答えを支えているか」を Noul 1 問。0.5 未満を誤りとする
 * - llm:<model>: 同じ問いを claude -p に true / false で答えさせる
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TypeSafeClient } from "@typesafe-ai/sdk";
import { JEV_MODEL, jevCostUsd } from "../harness/jev.js";
import { callClaude, measureOverhead } from "./lib/claude-cli.js";
import { callDecisions, noulAsPredicate, requireDecisionsKey } from "./lib/decisions-api.js";

const DIR = "experiments/recipes/halueval";

type Item = { id: string; knowledge: string; question: string; answer: string; hallucinated: boolean };
const OTHER = process.argv.includes("jev-other");
const items: Item[] = readFileSync(join(DIR, "data/qa_data.json"), "utf8")
  .split("\n")
  .filter((l) => l.trim() !== "")
  .map((l, i) => {
    const r = JSON.parse(l) as { knowledge: string; question: string; right_answer: string; hallucinated_answer: string };
    const hallucinated = (i % 2 === 1) !== OTHER;
    return { id: String(i), knowledge: r.knowledge, question: r.question, answer: hallucinated ? r.hallucinated_answer : r.right_answer, hallucinated };
  });

const INSTRUCTIONS = "Is the answer to the question correct and supported by the knowledge? Judge only from the knowledge.";
const CRITERIA = {
  true: "The knowledge supports the answer, and the answer correctly answers the question.",
  false: "The answer contradicts the knowledge, is not supported by it, or does not answer the question correctly.",
};

const client = new TypeSafeClient({ defaultModel: JEV_MODEL });

async function judgeJev(item: Item): Promise<Record<string, unknown>> {
  const startedAt = Date.now();
  try {
    const result = await client.systemOne({
      state: { knowledge: item.knowledge, question: item.question, answer: item.answer },
      questions: { supported: { type: "noul", instructions: INSTRUCTIONS, criteria: CRITERIA } },
    });
    const p = (result.answers as Record<string, { noul: number }>).supported?.noul;
    return { ok: true, supported: p, predicted_hallucinated: (p ?? 1) < 0.5, model: result.model, input_tokens: result.usage.input_tokens, cost_usd: jevCostUsd(result.usage.input_tokens), duration_ms: Date.now() - startedAt };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error), duration_ms: Date.now() - startedAt };
  }
}

async function judgeLlm(item: Item, model: string): Promise<Record<string, unknown>> {
  const r = await callClaude({
    model,
    system: `${INSTRUCTIONS}\ntrue: ${CRITERIA.true}\nfalse: ${CRITERIA.false}`,
    schema: { type: "object", properties: { supported: { type: "boolean" } }, required: ["supported"] },
    input: JSON.stringify({ knowledge: item.knowledge, question: item.question, answer: item.answer }),
  });
  if (!r.ok) return r;
  const { output, ...rest } = r;
  return { ...rest, predicted_hallucinated: (output as { supported?: boolean }).supported !== true };
}

async function judgeDecisions(item: Item): Promise<Record<string, unknown>> {
  const result = await callDecisions(JSON.stringify({ knowledge: item.knowledge, question: item.question, answer: item.answer }), [
    noulAsPredicate("supported", INSTRUCTIONS, CRITERIA),
  ]);
  if (!result.ok) return result;
  const answer = result.answers.supported;
  if (answer?.type !== "predicate") return { ...result, ok: false, error: "unexpected supported answer" };
  return { ...result, answers: undefined, supported: answer.probability,
    predicted_hallucinated: answer.probability < 0.5 };
}

function sample(all: Item[], n: number): Item[] {
  const step = all.length / n;
  // 偶数と奇数（正しい答えと誤った答え）が半々になるよう、等間隔で抜いたあと隣も取る
  return Array.from({ length: n / 2 }, (_, k) => Math.floor(k * step * 2) & ~1).flatMap((i) => [all[i], all[i + 1]]).filter((i): i is Item => i !== undefined);
}

async function main() {
  const [mode = "", ...rest] = process.argv.slice(2).filter((a) => a !== "--");
  if (mode.startsWith("overhead:")) {
    const model = mode.slice(9);
    writeFileSync(join(DIR, `overhead-${model}.json`), JSON.stringify({ model, ...await measureOverhead(model) }) + "\n");
    return;
  }
  if (mode === "report") {
    // 方式ごとに失敗行を除くと、方式間で採点する件がずれる。
    // 「LLM と同じ件」は、全方式が成功した ID だけに揃え、正解ラベルも方式間で照合する。
    type Row = { id: string; hallucinated: boolean; result: { ok?: boolean; predicted_hallucinated?: boolean; supported?: number } };
    const read = (m: string): Row[] => readFileSync(join(DIR, `results-${m}.jsonl`), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as Row);
    const modes = ["length", "jev", "decisions", "llm-haiku", "llm-sonnet"].filter((m) => existsSync(join(DIR, `results-${m}.jsonl`)));
    const all = new Map(modes.map((m) => [m, read(m)]));
    const okRows = new Map(modes.map((m) => [m, new Map((all.get(m) ?? []).filter((r) => r.result.ok !== false).map((r) => [r.id, r]))]));
    const haiku = okRows.get("llm-haiku");
    const common = haiku === undefined ? undefined : [...haiku.keys()].filter((id) => modes.every((m) => okRows.get(m)?.has(id)));
    for (const id of common ?? []) {
      const gold = okRows.get("jev")?.get(id)?.hallucinated;
      if (modes.some((m) => okRows.get(m)?.get(id)?.hallucinated !== gold)) throw new Error(`正解ラベルが方式間で異なる: ${id}`);
    }
    const acc = (rs: Row[]) => `${((100 * rs.filter((r) => r.result.predicted_hallucinated === r.hallucinated).length) / rs.length).toFixed(1)}%（${rs.length} 件）`;
    for (const m of modes) {
      const rows = [...(okRows.get(m)?.values() ?? [])];
      const failed = (all.get(m)?.length ?? 0) - rows.length;
      const same = common === undefined ? "" : `、LLM と同じ件 ${acc(common.map((id) => okRows.get(m)?.get(id)).filter((r): r is Row => r !== undefined))}`;
      console.log(`${m}: 全体 ${acc(rows)}${failed > 0 ? `（失敗 ${failed} 件を除外）` : ""}${same}`);
    }
    // 同じ件どうしの正誤の食い違いで、差が偶然でも起こりうるかを確かめる（McNemar の正確検定、両側）
    const mcnemar = (a: string, b: string) => {
      let aOnly = 0;
      let bOnly = 0;
      for (const id of common ?? []) {
        const ra = okRows.get(a)?.get(id);
        const rb = okRows.get(b)?.get(id);
        const ca = ra?.result.predicted_hallucinated === ra?.hallucinated;
        const cb = rb?.result.predicted_hallucinated === rb?.hallucinated;
        if (ca && !cb) aOnly++;
        if (cb && !ca) bOnly++;
      }
      const n = aOnly + bOnly;
      const k = Math.min(aOnly, bOnly);
      let tail = 0;
      let c = 1;
      for (let i = 0; i <= n; i++) {
        if (i > 0) c = (c * (n - i + 1)) / i;
        if (i <= k) tail += c / 2 ** n;
      }
      console.log(`検定 ${a} と ${b}: ${a} だけ正解 ${aOnly}、${b} だけ正解 ${bOnly}、p=${Math.min(1, 2 * tail).toFixed(4)}`);
    };
    for (const [a, b] of [["length", "jev"], ["length", "decisions"], ["length", "llm-haiku"], ["length", "llm-sonnet"],
      ["jev", "decisions"], ["jev", "llm-haiku"], ["jev", "llm-sonnet"]]) {
      if (common !== undefined && modes.includes(a!) && modes.includes(b!)) mcnemar(a!, b!);
    }
    for (const m of ["jev", "decisions"]) {
      const rows = (common ?? []).map((id) => okRows.get(m)?.get(id)?.result as { cost_usd?: number; duration_ms?: number } | undefined);
      const costs = rows.map((r) => r?.cost_usd).filter((v): v is number => typeof v === "number");
      const durations = rows.map((r) => r?.duration_ms).filter((v): v is number => typeof v === "number").sort((x, y) => x - y);
      if (costs.length === rows.length && durations.length === rows.length) {
        console.log(`${m} の同じ件: 1,000件あたり $${((costs.reduce((x, y) => x + y, 0) / costs.length) * 1000).toFixed(4)}、待ち中央値 ${durations[Math.floor(durations.length / 2)]?.toFixed(0)}ms`);
      }
    }
    if (existsSync(join(DIR, "results-jev-other.jsonl"))) {
      const a = okRows.get("jev") ?? new Map<string, Row>();
      const b = new Map(read("jev-other").filter((r) => r.result.ok !== false).map((r) => [r.id, r]));
      let pairs = 0;
      let right = 0;
      let ties = 0;
      for (const [id, x] of a) {
        const y = b.get(id);
        if (y === undefined || x.result.supported === undefined || y.result.supported === undefined) continue;
        pairs++;
        const [good, bad] = x.hallucinated ? [y, x] : [x, y];
        if ((good.result.supported ?? 0) > (bad.result.supported ?? 0)) right++;
        else if (good.result.supported === bad.result.supported) ties++;
      }
      console.log(`jev の対の比較: 正しい答えを高く見た ${right}/${pairs}（${((100 * right) / pairs).toFixed(1)}%）、同点 ${ties}`);
      const raw = readFileSync(join(DIR, "data/qa_data.json"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as { right_answer: string; hallucinated_answer: string });
      const shorter = raw.filter((r) => r.right_answer.split(/\s+/).length < r.hallucinated_answer.split(/\s+/).length).length;
      console.log(`length の対の比較: 正しい答えのほうが短い ${shorter}/${raw.length}（${((100 * shorter) / raw.length).toFixed(1)}%）`);
    }
    return;
  }
  const sampleAt = rest.indexOf("--sample");
  const targets = sampleAt >= 0 ? sample(items, Number(rest[sampleAt + 1])) : items;
  if (mode === "decisions") requireDecisionsKey();
  const rows: unknown[] = [];
  const queue = [...targets];
  await Promise.all(
    Array.from({ length: mode === "decisions" || mode.startsWith("llm:") ? 3 : 8 }, async () => {
      for (let item = queue.shift(); item !== undefined; item = queue.shift()) {
        let result: Record<string, unknown>;
        if (mode === "length") result = { ok: true, predicted_hallucinated: item.answer.split(/\s+/).length > 5 };
        else if (mode === "jev" || mode === "jev-other") result = await judgeJev(item);
        else if (mode === "decisions") result = await judgeDecisions(item);
        else if (mode.startsWith("llm:")) result = await judgeLlm(item, mode.slice(4));
        else throw new Error(`方式が分からない: ${mode}`);
        rows.push({ id: item.id, hallucinated: item.hallucinated, result });
      }
    }),
  );
  writeFileSync(join(DIR, `results-${mode.replace(":", "-")}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  console.log(`${mode}: ${rows.length} 件`);
}

await main();
