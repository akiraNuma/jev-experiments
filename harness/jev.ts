/**
 * Jev への問い合わせ。ハーネスから使う。アプリ本体からは呼ばない。
 *
 * 呼び出し結果は必ず ok で分岐する。
 * **API 障害と「Jev がその判断をした」を混ぜない。** 混ぜると、
 * 障害が起きたときに「確率が低かった」と誤読して実験結果が壊れる。
 */
import { TypeSafeClient, noul } from "@typesafe-ai/sdk";
import type { EntryType } from "@typesafe-ai/sdk";

/**
 * 測定で使う Jev の版。省略すると SDK は jev-latest を使い、新しい版が出ると答えが変わる。
 * https://docs.typesafe.ai/models の Model aliases を参照。
 */
export const JEV_MODEL = "jev-1.13.0";

const client = new TypeSafeClient({ defaultModel: JEV_MODEL });

/**
 * 入力トークンの単価（USD / トークン）。出力トークンは無料。
 * https://docs.typesafe.ai/models — Jev 1.13 は $0.042 / Mtok。
 * モデルが変われば単価も変わるので、応答の model と一緒に記録すること。
 */
const USD_PER_INPUT_TOKEN = 0.042 / 1_000_000;

export function jevCostUsd(inputTokens: number): number {
  return inputTokens * USD_PER_INPUT_TOKEN;
}

export type NoulAsk = {
  state: EntryType;
  instructions: string;
  criteria: { true: string; false: string };
};

export type NoulOutcome =
  | {
      ok: true;
      /** yes である確率。0〜1 */
      probability: number;
      model: string;
      usage: { input_tokens: number; output_tokens: number };
      /** 入力トークンから算出。出力は無料なので含めない */
      cost_usd: number;
      duration_ms: number;
    }
  | {
      ok: false;
      /** API 側の失敗。Jev の判断ではない */
      error: string;
      duration_ms: number;
    };

export async function askNoul(ask: NoulAsk): Promise<NoulOutcome> {
  const startedAt = Date.now();

  try {
    const result = await client.systemOne({
      state: ask.state,
      questions: {
        // noul(instructions, criteria)。criteria は yes / no それぞれの意味の説明
        verdict: noul(ask.instructions, { true: ask.criteria.true, false: ask.criteria.false }),
      },
    });

    return {
      ok: true,
      probability: result.answers.verdict.noul,
      model: result.model,
      usage: result.usage,
      cost_usd: jevCostUsd(result.usage.input_tokens),
      duration_ms: Date.now() - startedAt,
    };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
      duration_ms: Date.now() - startedAt,
    };
  }
}
