/** APIを呼ばず、比較用の送信形式・応答検査・料金集計を確かめる。 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { callDecisions, noulAsPredicate } from "./lib/decisions-api.js";

const originalFetch = globalThis.fetch;
const originalKey = process.env.OPENAI_API_KEY;
process.env.OPENAI_API_KEY = "test-key";

try {
  let body: Record<string, unknown> = {};
  globalThis.fetch = (_url, options) => {
    const bodyValue = options?.body;
    assert.equal(typeof bodyValue, "string");
    body = JSON.parse(bodyValue as string) as Record<string, unknown>;
    assert.equal((options?.headers as Record<string, string>).Authorization, "Bearer test-key");
    return Promise.resolve(new Response(JSON.stringify({
      model: "gpt-6-luna",
      answers: [{ type: "choice", name: "route", choice: "billing", confidence: 0.9,
        probabilities: [{ value: "billing", probability: 0.9 }, { value: "other", probability: 0.1 }] }],
      usage: { input_tokens: 2000, input_tokens_details: { cached_tokens: 1000, cache_write_tokens: 500 } },
    }), { status: 200 }));
  };
  const result = await callDecisions("refund", [{ type: "choice", name: "route", instructions: "route it",
    choices: [{ value: "billing" }, { value: "other" }] }]);
  assert.equal(body.model, "gpt-6-luna");
  assert.equal(body.input, "refund");
  assert.equal(result.ok, true);
  if (result.ok) {
    assert.equal(result.answers.route?.type, "choice");
    assert.equal(result.input_tokens, 2000);
    assert.equal(result.request_sha256, createHash("sha256").update(JSON.stringify(body)).digest("hex"));
    assert.equal(result.cost_usd, (500 * 0.10 + 1000 * 0.01 + 500 * 0.125) / 1_000_000);
  }

  globalThis.fetch = () => Promise.resolve(new Response(JSON.stringify({ model: "gpt-6-luna", answers: [],
    usage: { input_tokens: 1 } }), { status: 200 }));
  assert.equal((await callDecisions("x", [{ type: "predicate", name: "p", instructions: "p" }])).ok, false);

  // 不正な分布・確信度・利用量が成功に化けないこと。
  const question = { type: "choice" as const, name: "route", instructions: "route it",
    choices: [{ value: "billing" }, { value: "other" }] };
  const valid = { model: "gpt-6-luna", answers: [{ type: "choice", name: "route", choice: "billing", confidence: 0.9,
    probabilities: [{ value: "billing", probability: 0.9 }, { value: "other", probability: 0.1 }] }],
    usage: { input_tokens: 2000, input_tokens_details: { cached_tokens: 1000, cache_write_tokens: 500 } } };
  const invalid: unknown[] = [
    { ...valid, model: "wrong-model" },
    { ...valid, answers: [{ ...valid.answers[0], confidence: 2 }] },
    { ...valid, answers: [{ ...valid.answers[0], probabilities: [] }] },
    { ...valid, answers: [{ ...valid.answers[0], probabilities: [
      { value: "billing", probability: -0.1 }, { value: "other", probability: 1.1 }] }] },
    { ...valid, answers: [{ ...valid.answers[0], probabilities: [
      { value: "billing", probability: 0.5 }, { value: "billing", probability: 0.5 }] }] },
    { ...valid, answers: [{ ...valid.answers[0], probabilities: [
      { value: "billing", probability: 0.1 }, { value: "other", probability: 0.1 }] }] },
    { ...valid, usage: { input_tokens: 1.5 } },
    { ...valid, usage: { input_tokens: 2000, input_tokens_details: { cached_tokens: "oops" } } },
    { ...valid, usage: { input_tokens: 2000, input_tokens_details: { cache_write_tokens: 3000 } } },
  ];
  for (const payload of invalid) {
    globalThis.fetch = () => Promise.resolve(new Response(JSON.stringify(payload), { status: 200 }));
    assert.equal((await callDecisions("x", [question])).ok, false);
  }
  globalThis.fetch = () => Promise.resolve(new Response(JSON.stringify({ ...valid,
    answers: [{ type: "refusal", name: "route" }] }), { status: 200 }));
  const refused = await callDecisions("x", [question]);
  assert.equal(refused.ok && refused.answers.route?.type, "refusal");
  assert.deepEqual(refused.ok && refused.refused_questions, ["route"]);
  const rubric = { type: "score" as const, name: "severity", instructions: "rate it",
    levels: [{ label: "low" }, { label: "high" }] };
  const score = { type: "score", name: "severity", score: 0.8, confidence: 0.5,
    probabilities: [{ value: 0, label: "low", probability: 0.2 }, { value: 1, label: "high", probability: 0.8 }] };
  for (const [answer, expected] of [[score, true], [{ ...score, score: 2 }, false],
    [{ ...score, probabilities: [] }, false], [{ ...score, score: 0.2 }, false]] as const) {
    globalThis.fetch = () => Promise.resolve(new Response(JSON.stringify({ ...valid, answers: [answer] }), { status: 200 }));
    assert.equal((await callDecisions("x", [rubric])).ok, expected);
  }

  globalThis.fetch = () => Promise.resolve(new Response(JSON.stringify({ error: { type: "rate_limit" } }), { status: 429 }));
  const limited = await callDecisions("x", [{ type: "predicate", name: "p", instructions: "p" }]);
  assert.equal(limited.ok, false);
  assert.equal(limited.status, 429);
  assert.match(limited.request_sha256, /^[a-f0-9]{64}$/);
} finally {
  globalThis.fetch = originalFetch;
  if (originalKey === undefined) delete process.env.OPENAI_API_KEY;
  else process.env.OPENAI_API_KEY = originalKey;
}

// Noul の条件は、Jev と同じ文言のまま True:/False: の見出しを付けて並べる
assert.deepEqual(noulAsPredicate("ok", "Is it ok?", { true: "Yes case.", false: "No case." }),
  { type: "predicate", name: "ok", instructions: "Is it ok?\nTrue: Yes case.\nFalse: No case." });
assert.deepEqual(noulAsPredicate("ok", "Is it ok?"), { type: "predicate", name: "ok", instructions: "Is it ok?" });

console.log("decisions API adapter: 19 checks passed");
