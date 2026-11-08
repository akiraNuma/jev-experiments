/** OpenAI Decisions API の比較用入口。既存の Jev 測定からは呼ばない。 */

import { createHash } from "node:crypto";

export const DECISIONS_MODEL = "gpt-6-luna";

type TextPart = { type: "input_text"; text: string };
type ImagePart = { type: "input_image"; image_url: string };
type Part = TextPart | ImagePart;
export type DecisionInput = string | { role: "user"; content: Part[] }[];
export type DecisionQuestion =
  | { type: "predicate"; name: string; instructions: string | Part[] }
  | { type: "choice"; name: string; instructions: string | Part[]; choices: { value: string; description?: string }[] }
  | { type: "score"; name: string; instructions: string | Part[]; levels: { label: string; description?: string }[] };

export type DecisionAnswer =
  | { type: "predicate"; name: string; probability: number }
  | { type: "choice"; name: string; choice: string; confidence: number; probabilities: { value: string; probability: number }[] }
  | { type: "score"; name: string; score: number; confidence: number; probabilities: { value: number; label: string; probability: number }[] }
  | { type: "refusal"; name: string };

type DecisionResponse = {
  model: string;
  answers: DecisionAnswer[];
  usage: { input_tokens: number; input_tokens_details?: { cached_tokens?: number; cache_write_tokens?: number } };
};

/**
 * JevのNoul（質問文＋true/false の条件）を、Decisions APIのpredicateに写す。
 * Decisions APIのpredicateには条件の項目がないため、質問文の後ろに見出し付きで並べる。
 * どの実験もこの形にそろえ、相手にだけ文を足したり言い換えたりしない。
 */
export function noulAsPredicate(name: string, instructions: string, criteria?: { true?: string; false?: string }): DecisionQuestion {
  const lines = [instructions];
  if (criteria?.true !== undefined) lines.push(`True: ${criteria.true}`);
  if (criteria?.false !== undefined) lines.push(`False: ${criteria.false}`);
  return { type: "predicate", name, instructions: lines.join("\n") };
}

export function requireDecisionsKey(): string {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY is not configured; set it in .env and run Node with --env-file=.env");
  return key;
}

function probability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

export async function callDecisions(input: DecisionInput, questions: DecisionQuestion[]) {
  const key = requireDecisionsKey();
  const body = JSON.stringify({ model: DECISIONS_MODEL, input, questions });
  const requestSha256 = createHash("sha256").update(body).digest("hex");
  const started = performance.now();
  let status: number | undefined;
  try {
    const response = await fetch("https://api.openai.com/v1/decisions", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body,
      signal: AbortSignal.timeout(60_000),
    });
    status = response.status;
    const payload: unknown = await response.json();
    if (!response.ok) return { ok: false as const, status, request_sha256: requestSha256, error: `Decisions API HTTP ${status}`, duration_ms: performance.now() - started };
    const data = payload as DecisionResponse;
    if (data.model !== DECISIONS_MODEL || !Array.isArray(data.answers) || !Number.isSafeInteger(data.usage?.input_tokens)) {
      throw new Error("Decisions API response has an unexpected shape");
    }
    const answers = Object.fromEntries(data.answers.map((answer) => [answer.name, answer]));
    if (data.answers.length !== questions.length || questions.some((question) => !Object.hasOwn(answers, question.name))) {
      throw new Error("Decisions API returned missing answers");
    }
    for (const question of questions) {
      const answer = answers[question.name];
      if (answer?.type === "refusal") continue;
      if (answer?.type !== question.type) throw new Error(`Decisions API returned wrong type for ${question.name}`);
      if (answer.type === "predicate" && (!probability(answer.probability))) {
        throw new Error(`Decisions API returned invalid probability for ${question.name}`);
      }
      if (answer.type === "choice" && question.type === "choice" && (!question.choices.some((choice) => choice.value === answer.choice)
        || !probability(answer.confidence) || !Array.isArray(answer.probabilities)
        || answer.probabilities.length !== question.choices.length
        || new Set(answer.probabilities.map((entry) => entry.value)).size !== question.choices.length
        || answer.probabilities.some((entry) => !question.choices.some((choice) => choice.value === entry.value)
          || !probability(entry.probability))
        || Math.abs(answer.probabilities.reduce((sum, entry) => sum + entry.probability, 0) - 1) > 0.001)) {
        throw new Error(`Decisions API returned invalid choice for ${question.name}`);
      }
      if (answer.type === "score" && question.type === "score" && (!Number.isFinite(answer.score)
        || answer.score < 0 || answer.score > question.levels.length - 1 || !probability(answer.confidence)
        || !Array.isArray(answer.probabilities) || answer.probabilities.length !== question.levels.length
        || new Set(answer.probabilities.map((entry) => entry.value)).size !== question.levels.length
        || answer.probabilities.some((entry) => !Number.isInteger(entry.value) || entry.value < 0
          || entry.value >= question.levels.length || question.levels[entry.value]?.label !== entry.label
          || !probability(entry.probability))
        || Math.abs(answer.probabilities.reduce((sum, entry) => sum + entry.probability, 0) - 1) > 0.001
        || Math.abs(answer.probabilities.reduce((sum, entry) => sum + entry.value * entry.probability, 0) - answer.score) > 0.001)) {
        throw new Error(`Decisions API returned invalid score for ${question.name}`);
      }
    }
    const inputTokens = data.usage.input_tokens;
    const cachedTokens = data.usage.input_tokens_details?.cached_tokens ?? 0;
    const cacheWriteTokens = data.usage.input_tokens_details?.cache_write_tokens ?? 0;
    if (![inputTokens, cachedTokens, cacheWriteTokens].every((value) => Number.isSafeInteger(value) && value >= 0)
      || cachedTokens + cacheWriteTokens > inputTokens) {
      throw new Error("Decisions API returned invalid token usage");
    }
    // 2026-11-01 の標準・短文脈料金。長文脈や地域処理の加算はここに含めない。
    // https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability
    // https://developers.openai.com/api/docs/pricing
    const estimatedCostUsd = ((inputTokens - cachedTokens - cacheWriteTokens) * 0.10
      + cachedTokens * 0.01 + cacheWriteTokens * 0.125) / 1_000_000;
    return {
      ok: true as const, status, model: data.model, answers, request_sha256: requestSha256,
      refused_questions: data.answers.filter((answer) => answer.type === "refusal").map((answer) => answer.name),
      input_tokens: inputTokens, cached_input_tokens: cachedTokens, cache_write_tokens: cacheWriteTokens,
      cost_usd: estimatedCostUsd,
      duration_ms: performance.now() - started,
    };
  } catch (error) {
    return { ok: false as const, status, request_sha256: requestSha256, error: error instanceof Error ? error.message : String(error), duration_ms: performance.now() - started };
  }
}
