/**
 * 【計測用】レシピの測定で、LLM を `claude -p` で呼ぶ共通部分。
 *
 * Anthropic の API キーは使わない（著者の判断、2026-10-14）。`claude -p` は 1 回ごとに Claude Code の前置きを
 * 数万トークン送るので、記録の費用はそのまま使えない。そこで:
 * - 呼び出しごとに、入力トークンの合計、出力トークン、そのうち考える段のトークンを残す
 * - 最小の入力で呼んだときの入力トークン（前置き）を measureOverhead で測り、差し引く
 * - 差し引いた入力トークンと公式の単価から、API を直接呼んだときの費用を apiCostUsd で計算する（F-030）
 *
 * 前置きの量は作業ディレクトリの CLAUDE.md で変わるので、空のディレクトリで動かす。
 */
import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export const EMPTY_CWD = join(tmpdir(), "jev-recipes-empty");
mkdirSync(EMPTY_CWD, { recursive: true });

// 測定のエージェントと同じく、Jev のキーや Claude のセッション変数を子に渡さない
/**
 * ユーザー階層の設定（~/.claude/CLAUDE.md・settings・MCP）を読ませない。
 * 指定しないと、測定する人の個人設定が毎回の入力に入り、判定にも影響しうる（2026-11-02に確認）。
 */
export const ISOLATION_ARGS = ["--setting-sources", "project,local", "--strict-mcp-config"];

const CHILD_ENV = Object.fromEntries(Object.entries(process.env).filter(([k]) => !/^(CLAUDE|TYPESAFE_)/.test(k)));

/** 更新されるCLIの別名を、測定対象のモデルIDへ固定する。 */
export function claudeModelId(model: string): string {
  const models: Record<string, string> = { haiku: "claude-haiku-5-5", sonnet: "claude-sonnet-5-5" };
  return models[model] ?? model;
}

/** 応答にモデル情報がない場合は空配列にし、指定名から推定しない。 */
export function claudeModelsUsed(modelUsage: unknown): string[] {
  return modelUsage !== null && typeof modelUsage === "object" && !Array.isArray(modelUsage)
    ? Object.keys(modelUsage) : [];
}

export type ClaudeResult =
  | {
      ok: true;
      output: unknown;
      requested_model: string;
      models_used: string[];
      model_usage: unknown;
      input_total: number;
      output_tokens: number;
      thinking_tokens: number;
      duration_ms: number;
      duration_api_ms: number;
      cli_cost_usd: number;
    }
  | { ok: false; error: string; duration_ms: number };

type Usage = {
  input_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
  output_tokens: number;
  output_tokens_details?: { thinking_tokens?: number };
};

export function callClaude(opts: { model: string; system: string; schema: object; input: string }): Promise<ClaudeResult> {
  const startedAt = Date.now();
  const requestedModel = claudeModelId(opts.model);
  return new Promise((resolve) => {
    const child = spawn(
      "claude",
      ["-p", "--model", requestedModel, ...ISOLATION_ARGS, "--tools", "", "--system-prompt", opts.system, "--json-schema", JSON.stringify(opts.schema), "--output-format", "json"],
      { env: CHILD_ENV, cwd: EMPTY_CWD, stdio: ["pipe", "pipe", "pipe"] },
    );
    let out = "";
    child.stdout.on("data", (d: Buffer) => (out += d.toString()));
    child.stdin.end(opts.input);
    child.on("close", () => {
      try {
        const parsed = JSON.parse(out) as {
          structured_output?: unknown;
          usage?: Usage;
          total_cost_usd?: number;
          duration_api_ms?: number;
          is_error?: boolean;
          result?: string;
          modelUsage?: unknown;
        };
        if (parsed.is_error === true || parsed.structured_output === undefined || parsed.usage === undefined) {
          resolve({ ok: false, error: String(parsed.result ?? "no output").slice(0, 200), duration_ms: Date.now() - startedAt });
          return;
        }
        const u = parsed.usage;
        resolve({
          ok: true,
          output: parsed.structured_output,
          requested_model: requestedModel,
          models_used: claudeModelsUsed(parsed.modelUsage),
          model_usage: parsed.modelUsage ?? null,
          input_total: u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens,
          output_tokens: u.output_tokens,
          thinking_tokens: u.output_tokens_details?.thinking_tokens ?? 0,
          duration_ms: Date.now() - startedAt,
          duration_api_ms: parsed.duration_api_ms ?? NaN,
          cli_cost_usd: parsed.total_cost_usd ?? NaN,
        });
      } catch {
        resolve({ ok: false, error: out.slice(0, 200), duration_ms: Date.now() - startedAt });
      }
    });
  });
}

/** 最小の入力で3回測り、前置きの中央値と各呼び出しのモデル情報を返す。 */
export async function measureOverhead(model: string) {
  const schema = { type: "object", properties: { a: { type: "string", enum: ["a"] } }, required: ["a"] };
  const totals: number[] = [];
  const samples: ClaudeResult[] = [];
  for (let i = 0; i < 3; i++) {
    const r = await callClaude({ model, system: "x", schema, input: "x" });
    samples.push(r);
    if (r.ok) totals.push(r.input_total);
  }
  if (totals.length === 0) throw new Error("前置きを測れなかった");
  return {
    tokens: totals.sort((a, b) => a - b)[Math.floor(totals.length / 2)] ?? NaN,
    requested_model: claudeModelId(model),
    models_used: [...new Set(samples.flatMap((r) => r.ok ? r.models_used : []))],
    samples,
  };
}

/**
 * 100 万トークンあたりの単価（USD）。2026-10-13 に公式の料金表で確認した
 * https://platform.claude.com/docs/en/about-claude/pricing
 * Haiku 5.5 はプロンプト 10 万トークン以下の単価。
 */
const PRICES: Record<string, { input: number; cacheRead: number; output: number }> = {
  haiku: { input: 0.1, cacheRead: 0.01, output: 0.5 },
  sonnet: { input: 2, cacheRead: 0.1, output: 10 },
};

/** 構造化出力の道具の前置き（料金表の tool use system prompt、Haiku 5.5 と Sonnet 5.5 の any/tool 指定時） */
export const TOOL_PROMPT_TOKENS = 406;

/**
 * API を直接呼んだときの費用。考える段は切る前提で、出力は考える段を除いた分だけ数える。
 * cachedTokens: 毎回同じ前半（system と選択肢の一覧など）のうち、キャッシュから読める分
 */
export function apiCostUsd(model: string, promptTokens: number, outputTokens: number, cachedTokens = 0): number {
  const p = PRICES[model];
  if (p === undefined) throw new Error(`単価が分からない: ${model}`);
  const uncached = promptTokens + TOOL_PROMPT_TOKENS - cachedTokens;
  return (uncached * p.input + cachedTokens * p.cacheRead + outputTokens * p.output) / 1_000_000;
}
