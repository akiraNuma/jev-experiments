#!/usr/bin/env tsx
/**
 * 【計測用】実験7：文から値を抜き出す（候補はコードで拾い、Jev に選ばせる）。
 *
 *   node --import tsx scripts/recipe-extract.ts first | jev | llm:haiku | overhead:haiku | report
 *
 * データ: CUAD v1（The Atticus Project、CC BY 4.0）の test の「契約日（Agreement Date）」。契約書 102 通のうち 93 通に、
 * 弁護士などが契約日の箇所を付けた。残り 9 通は契約日が書かれていない。
 *
 * 公式のクックブック（pre_parsed_value_extraction）と同じく、値を生成させない。
 * 1. コードが正規表現で日付らしい箇所を全部拾う（前後 150 字を添える）
 * 2. どれが契約日かを選ばせる。「どれでもない」も選べる
 *
 * 方式:
 * - first: 最初に出てくる日付を選ぶ。よくある素朴な規則
 * - jev: 候補を c0、c1 … として Choice 1 問。「none」を足す
 * - llm:<model>: 同じ候補の一覧を claude -p に渡し、番号か none を選ばせる
 *
 * 正解の判定は 2 通り。契約日が無い契約で none を選べば、どちらでも正解。
 * - 位置: 選んだ候補の位置が、正解の箇所と重なれば正解
 * - 値: 選んだ候補の日付が、正解の箇所の日付と同じ日なら正解。契約書には同じ契約日が何度も出てくる（表紙と本文の冒頭など）が、
 *   正解の箇所はそのうち 1 つにしか付いていない。位置だけで数えると、同じ日付の別の箇所を選んだ回を誤りにしてしまう
 * 正解の箇所と重なる候補が 1 つも無い契約は、拾い方の失敗として別に数える。
 *
 * report は、Jev の確信度が線より低い契約だけ Haiku の答えに置き換えた場合（カスケード、実験8）も出す。
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TypeSafeClient } from "@typesafe-ai/sdk";
import { JEV_MODEL, jevCostUsd } from "../harness/jev.js";
import { callClaude, measureOverhead } from "./lib/claude-cli.js";
import { callDecisions, requireDecisionsKey } from "./lib/decisions-api.js";

const DIR = "experiments/recipes/cuad";
const WINDOW = 150;
const MAX_CANDIDATES = 40;

type Doc = { id: string; context: string; answers: { text: string; answer_start: number }[] };
type Candidate = { start: number; end: number; text: string; snippet: string };

const docs: Doc[] = (JSON.parse(readFileSync(join(DIR, "data/agreement-date-test.json"), "utf8")) as { title: string; context: string; answers: Doc["answers"] }[]).map(
  (d) => ({ id: d.title, context: d.context, answers: d.answers }),
);

const MONTH = "(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\\.?|Feb\\.?|Mar\\.?|Apr\\.?|Jun\\.?|Jul\\.?|Aug\\.?|Sept?\\.?|Oct\\.?|Nov\\.?|Dec\\.?)";
const DATE_PATTERNS = [
  `${MONTH}\\s+\\d{1,2}(?:st|nd|rd|th)?\\s*,?\\s*\\d{4}`, // March 1, 2020
  `\\d{1,2}(?:st|nd|rd|th)?\\s+(?:day\\s+of\\s+)?${MONTH}\\s*,?\\s*\\d{4}`, // 1st day of March, 2020 / 1 March 2020
  `\\d{1,2}/\\d{1,2}/\\d{2,4}`, // 03/01/2020
  `${MONTH}\\s*,?\\s*\\d{4}`, // March 2020
];

/** 日付らしい箇所を拾う。重なる箇所は長いほうを残す */
function candidatesOf(text: string): Candidate[] {
  const found: { start: number; end: number }[] = [];
  for (const p of DATE_PATTERNS) {
    for (const m of text.matchAll(new RegExp(p, "gi"))) found.push({ start: m.index, end: m.index + m[0].length });
  }
  found.sort((a, b) => a.start - b.start || b.end - a.end);
  const kept: { start: number; end: number }[] = [];
  for (const f of found) {
    const last = kept[kept.length - 1];
    if (last !== undefined && f.start < last.end) continue;
    kept.push(f);
  }
  return kept.slice(0, MAX_CANDIDATES).map((k) => ({
    ...k,
    text: text.slice(k.start, k.end),
    snippet: text.slice(Math.max(0, k.start - WINDOW), k.end + WINDOW).replace(/\s+/g, " "),
  }));
}

/** 日付の書き方の違いをならして YYYY-MM-DD にする。読めなければ小文字にしただけの文字列を返す */
function normalizeDate(raw: string): string {
  const s = raw.replace(/(\d)(st|nd|rd|th)/gi, "$1").replace(/day\s+of\s+/i, "").replace(/\./g, "").replace(/Sept/i, "Sep").replace(/\s+/g, " ").trim();
  const slash = s.match(/^(\d{1,2})\/(\d{1,2})\/(\d{2,4})$/);
  if (slash) {
    const year = Number(slash[3]) < 100 ? Number(slash[3]) + 2000 : Number(slash[3]);
    return `${year}-${slash[1]?.padStart(2, "0")}-${slash[2]?.padStart(2, "0")}`;
  }
  const d = new Date(`${s.replace(/^(\d{1,2}) (\w+),? (\d{4})$/, "$2 $1, $3")} UTC`);
  return Number.isNaN(d.getTime()) ? s.toLowerCase() : d.toISOString().slice(0, 10);
}

/** 値で数えた正解。位置で正解なら正解、そうでなければ選んだ日付と正解の日付が同じ日かを見る */
function valueCorrect(doc: Doc, cands: Candidate[], predicted: string | undefined, correct: string[]): boolean {
  if (predicted !== undefined && correct.includes(predicted)) return true;
  if (doc.answers.length === 0 || predicted === undefined || predicted === "none") return false;
  const chosen = cands[Number(predicted.slice(1))];
  return chosen !== undefined && doc.answers.some((a) => normalizeDate(a.text) === normalizeDate(chosen.text));
}

function correctIds(doc: Doc, cands: Candidate[]): string[] {
  if (doc.answers.length === 0) return ["none"];
  return cands
    .map((c, i) => ({ c, i }))
    .filter(({ c }) => doc.answers.some((a) => c.start < a.answer_start + a.text.length && a.answer_start < c.end))
    .map(({ i }) => `c${i}`);
}

const INSTRUCTIONS =
  "Each candidate is a date found in a commercial contract, with the text around it. Which candidate is the date the contract was signed or entered into (the agreement date)? Choose none if no candidate is the agreement date.";

const client = new TypeSafeClient({ defaultModel: JEV_MODEL });

async function judgeJev(cands: Candidate[]): Promise<Record<string, unknown>> {
  const startedAt = Date.now();
  const criteria: Record<string, string> = Object.fromEntries(cands.map((c, i) => [`c${i}`, `"${c.text}" in: …${c.snippet}…`]));
  criteria.none = "None of the candidates is the agreement date.";
  try {
    const result = await client.systemOne({ state: { candidates: cands.length }, questions: { date: { type: "choice", instructions: INSTRUCTIONS, criteria } } });
    const a = (result.answers as Record<string, { choice: string; confidence: number }>).date;
    return { ok: true, predicted: a?.choice, confidence: a?.confidence, model: result.model, input_tokens: result.usage.input_tokens, cost_usd: jevCostUsd(result.usage.input_tokens), duration_ms: Date.now() - startedAt };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error), duration_ms: Date.now() - startedAt };
  }
}

async function judgeLlm(cands: Candidate[], model: string): Promise<Record<string, unknown>> {
  const ids = [...cands.map((_, i) => `c${i}`), "none"];
  const r = await callClaude({
    model,
    system: INSTRUCTIONS,
    schema: { type: "object", properties: { date: { type: "string", enum: ids } }, required: ["date"] },
    input: cands.map((c, i) => `c${i}: "${c.text}" in: …${c.snippet}…`).join("\n"),
  });
  if (!r.ok) return r;
  const { output, ...rest } = r;
  return { ...rest, predicted: (output as { date?: string }).date };
}

async function judgeDecisions(cands: Candidate[]): Promise<Record<string, unknown>> {
  const result = await callDecisions(JSON.stringify({ candidates: cands.length }), [{
    type: "choice", name: "date", instructions: INSTRUCTIONS,
    choices: [
      ...cands.map((candidate, i) => ({ value: `c${i}`, description: `"${candidate.text}" in: …${candidate.snippet}…` })),
      { value: "none", description: "None of the candidates is the agreement date." },
    ],
  }]);
  if (!result.ok) return result;
  const answer = result.answers.date;
  if (answer?.type !== "choice") return { ...result, ok: false, error: "unexpected date answer" };
  return { ...result, answers: undefined, predicted: answer.choice, confidence: answer.confidence };
}

async function main() {
  const [mode = ""] = process.argv.slice(2).filter((a) => a !== "--");
  if (mode.startsWith("overhead:")) {
    const model = mode.slice(9);
    writeFileSync(join(DIR, `overhead-${model}.json`), JSON.stringify({ model, ...await measureOverhead(model) }) + "\n");
    return;
  }
  if (mode === "report") {
    type Row = { id: string; n_candidates: number; correct: string[]; result: { ok?: boolean; predicted?: string; confidence?: number } };
    const byId = new Map(docs.map((d) => [d.id, d]));
    const load = (m: string): Row[] | undefined => {
      const path = join(DIR, `results-${m}.jsonl`);
      if (!existsSync(path)) return undefined;
      const rows = readFileSync(path, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as Row);
      // 候補 0 個の契約は、どの方式にも送れないので比較から外す（2026-11-06、本の実験7・8は98通で比べる）
      return rows.filter((r) => r.n_candidates > 0);
    };
    const score = (r: Row, predicted = r.result.predicted) => {
      const doc = byId.get(r.id);
      if (doc === undefined) throw new Error(`契約が見つからない: ${r.id}`);
      return { position: predicted !== undefined && r.correct.includes(predicted), value: valueCorrect(doc, candidatesOf(doc.context), predicted, r.correct) };
    };
    const pct = (n: number, all: number) => `${n}/${all}（${((100 * n) / all).toFixed(1)}%）`;
    for (const m of ["first", "jev", "decisions", "llm-haiku", "llm-sonnet"]) {
      const rows = load(m);
      if (rows === undefined) continue;
      const scores = rows.map((r) => score(r));
      console.log(`${m}: 位置で正解 ${pct(scores.filter((x) => x.position).length, rows.length)}、値で正解 ${pct(scores.filter((x) => x.value).length, rows.length)}、失敗 ${rows.filter((r) => r.result.ok === false).length}`);
    }
    // 片方だけが値で正解した数（McNemar の検定に使う）
    const jevRows = load("jev");
    if (jevRows !== undefined) {
      const jevRight = new Map(jevRows.map((r) => [r.id, score(r).value]));
      for (const m of ["first", "decisions", "llm-haiku", "llm-sonnet"]) {
        const rows = load(m);
        if (rows === undefined) continue;
        const onlyJev = rows.filter((r) => jevRight.get(r.id) === true && !score(r).value).length;
        const onlyOther = rows.filter((r) => jevRight.get(r.id) === false && score(r).value).length;
        console.log(`jev と ${m}: jev だけ正解 ${onlyJev}、${m} だけ正解 ${onlyOther}`);
      }
    }
    const dated = docs.filter((d) => d.answers.length > 0);
    const reachable = dated.filter((d) => {
      const cands = candidatesOf(d.context);
      return cands.some((_, i) => valueCorrect(d, cands, `c${i}`, correctIds(d, cands)));
    }).length;
    console.log(`契約日のある ${dated.length} 通のうち、正解の日付が候補にある契約: 位置で ${dated.filter((d) => correctIds(d, candidatesOf(d.context)).length > 0).length}、値で ${reachable}`);
    const haiku = new Map((load("llm-haiku") ?? []).map((r) => [r.id, r]));
    for (const firstMode of ["jev", "decisions"]) {
      const first = load(firstMode);
      if (first === undefined || haiku.size === 0) continue;
      if (first.some((r) => !haiku.has(r.id))) {
        console.log(`${firstMode} カスケード: Haiku の結果が不足。集計を保留`);
        continue;
      }
      for (const t of [0.5, 0.7, 0.9]) {
        let escalated = 0;
        let right = 0;
        for (const r of first) {
          const h = haiku.get(r.id);
          const useHaiku = r.n_candidates > 0 && (r.result.ok === false || (r.result.confidence ?? 0) < t) && h !== undefined;
          if (useHaiku) escalated++;
          if (score(r, useHaiku ? h.result.predicted : r.result.predicted).value) right++;
        }
        console.log(`${firstMode} カスケード 線 ${t}: Haiku に回した ${escalated}/${first.length}、値で正解 ${pct(right, first.length)}`);
      }
    }
    return;
  }
  const rows: unknown[] = [];
  if (mode === "decisions") requireDecisionsKey();
  const queue = [...docs];
  await Promise.all(
    Array.from({ length: mode === "decisions" || mode.startsWith("llm:") ? 3 : 8 }, async () => {
      for (let doc = queue.shift(); doc !== undefined; doc = queue.shift()) {
        const cands = candidatesOf(doc.context);
        const correct = correctIds(doc, cands);
        let result: Record<string, unknown>;
        // 候補が無ければ「どれでもない」しか無いので、どの方式でも聞かない
        if (cands.length === 0) result = { ok: true, predicted: "none", skipped: true };
        else if (mode === "first") result = { ok: true, predicted: cands.length > 0 ? "c0" : "none" };
        else if (mode === "jev") result = await judgeJev(cands);
        else if (mode === "decisions") result = await judgeDecisions(cands);
        else if (mode.startsWith("llm:")) result = await judgeLlm(cands, mode.slice(4));
        else throw new Error(`方式が分からない: ${mode}`);
        rows.push({ id: doc.id, n_candidates: cands.length, correct, result });
      }
    }),
  );
  writeFileSync(join(DIR, `results-${mode.replace(":", "-")}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  console.log(`${mode}: ${rows.length} 通`);
}

await main();
