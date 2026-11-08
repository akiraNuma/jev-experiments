#!/usr/bin/env tsx
/**
 * 【計測用】classify-judge.ts と classify-baseline.py の結果を集計する。
 *
 *   npx tsx scripts/classify-report.ts
 *
 * LLM は等間隔で抜いた 400 件だけを回したので、どの方式も「LLM と同じ 400 件」でも出す。
 * Jev は確信度で線を引き、線より上だけを自動で振り分けたときの、残る件数と正解率も出す。
 * 回答拒否は除かず不正解として数える。確信度がないので、線を引く集計では確認待ちの側に入れる（findings F-047）。
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const DIR = "experiments/classify";
const MODES = ["tfidf", "jev", "decisions", "llm-haiku", "llm-sonnet"];
const CONFIDENCE_LINES = [0.5, 0.7, 0.9];

type Row = {
  id: string;
  label: string;
  result: { ok?: boolean; error?: string; refused_questions?: string[]; predicted?: string; confidence?: number; top3?: [string, number][]; cost_usd?: number; duration_ms?: number };
};

function load(dataset: string, mode: string): Row[] | undefined {
  const path = join(DIR, `results-${dataset}-${mode}.jsonl`);
  if (!existsSync(path)) return undefined;
  const rows = readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.trim() !== "")
    .map((l) => JSON.parse(l) as Row);
  if (new Set(rows.map((row) => row.id)).size !== rows.length) throw new Error(`duplicate IDs: ${path}`);
  return rows;
}

function pct(n: number, d: number): string {
  return d === 0 ? "-" : `${((100 * n) / d).toFixed(1)}%（${n}/${d}）`;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? NaN;
}

const correct = (r: Row) => r.result.ok !== false && r.result.predicted === r.label;
// classify-judge.ts は Decisions API の refusal にこの誤りを付ける（decisions-api.ts は refusal 以外の想定外の型を例外にする）
const refused = (r: Row) => r.result.ok === false && (r.result.error === "unexpected intent answer" || (r.result.refused_questions?.length ?? 0) > 0);

for (const dataset of ["banking77", "massive-ja"]) {
  console.log(`\n## ${dataset}`);
  const sampleIds = new Set((load(dataset, "llm-haiku") ?? load(dataset, "llm-sonnet") ?? []).map((r) => r.id));
  for (const mode of MODES) {
    const rows = load(dataset, mode);
    if (rows === undefined) continue;
    const ok = rows.filter((r) => r.result.ok !== false);
    // 採点の対象：成功と回答拒否。通信失敗などは除いて件数を表示する
    const scored = rows.filter((r) => r.result.ok !== false || refused(r));
    const sample = rows.filter((r) => sampleIds.has(r.id));
    const inSample = sample.filter((r) => r.result.ok !== false);
    const allLabel = mode === "decisions" ? "測定分" : "全体";
    const sampleScored = sample.filter((r) => r.result.ok !== false || refused(r));
    console.log(`- ${mode}: ${allLabel} ${pct(scored.filter(correct).length, scored.length)}、共通対象 ${sample.length}/${sampleIds.size} 件中の正解 ${pct(sampleScored.filter(correct).length, sampleScored.length)}、回答拒否 ${rows.filter(refused).length} 件（不正解として数える）、それ以外の失敗 ${rows.length - scored.length} 件`);
    const timed = sampleIds.size > 0 ? inSample : ok;
    const costs = timed.map((r) => r.result.cost_usd).filter((v): v is number => typeof v === "number");
    const durations = timed.map((r) => r.result.duration_ms).filter((v): v is number => typeof v === "number");
    if (costs.length > 0) console.log(`  - 成功した共通対象 ${timed.length} 件の1,000 件あたり $${((1000 * costs.reduce((a, b) => a + b, 0)) / costs.length).toFixed(3)}、1 件の待ち 中央値 ${median(durations)}ms`);
    if (mode === "jev" || mode === "decisions") {
      const top3 = scored.filter((r) => r.result.ok !== false && r.result.top3?.some(([l]) => l === r.label)).length;
      console.log(`  - 上位 3 つに正解が入った ${pct(top3, scored.length)}`);
      for (const line of CONFIDENCE_LINES) {
        const sure = scored.filter((r) => r.result.ok !== false && (r.result.confidence ?? 0) >= line);
        console.log(`  - 確信度 ${line} 以上: ${pct(sure.length, scored.length)} を自動で振り分け、その正解率 ${pct(sure.filter(correct).length, sure.length)}`);
      }
    }
  }
}
