#!/usr/bin/env tsx
/**
 * 【計測用】画像の分類（文字/見本）と頭の位置。取得は prepare-pets.py。
 * node --import tsx scripts/recipe-images.ts <mode> <dev|eval|control>
 * mode: text-1.13 / text-1.14 / examples-1.13 / examples-1.14 / box-1.14 / decisions-text / decisions-examples
 * HTTP multipart を直接使う。既存レシピが使う SDK 0.6 の挙動を変えない。
 * 1画像1リクエスト、同時実行2、再試行なし。失敗も保存し、集計の分母に残す。
 */
import "dotenv/config";
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { callDecisions, DECISIONS_MODEL, requireDecisionsKey } from "./lib/decisions-api.js";
import type { DecisionQuestion } from "./lib/decisions-api.js";

const DIR = "experiments/recipes/pets";
type Box = { left: number; top: number; right: number; bottom: number };
type Item = { id: string; breed: string; species: string; split: string; sha256: string; box_784: Box };
const manifest = JSON.parse(readFileSync(join(DIR, "manifest.json"), "utf8")) as { items: Item[] };
const criteria: Record<string, string> = {
  British_Shorthair: "A British Shorthair cat.",
  Russian_Blue: "A Russian Blue cat.",
  Siamese: "A Siamese cat.",
  beagle: "A beagle dog.",
  boxer: "A boxer dog.",
  pug: "A pug dog.",
};
const BREED_INSTRUCTIONS = "Which of the six breeds is the main pet in `photo`?";
const BOX_INSTRUCTIONS = "The tight bounding box around the head of the main pet in `photo`, including ears but not the neck or body.";
const MODES = ["text-1.13", "text-1.14", "examples-1.13", "examples-1.14", "box-1.14", "decisions-text", "decisions-examples"];
const [mode = "", split = "dev"] = process.argv.slice(2);
if (!MODES.includes(mode) || !["dev", "eval", "control"].includes(split)) {
  throw new Error(`usage: recipe-images.ts <${MODES.join("|")}> <dev|eval|control>`);
}
const model = mode.endsWith("1.13") ? "jev-1.13.0" : "jev-1.14.0";
const isDecisions = mode.startsWith("decisions-");
const requestedModel = isDecisions ? DECISIONS_MODEL : model;
const isBox = mode === "box-1.14";
const key = process.env.TYPESAFE_API_KEY;
if (isDecisions) requireDecisionsKey();
else if (!key) throw new Error("TYPESAFE_API_KEY is not configured");
if (isDecisions && split === "control") throw new Error("control split is only for Jev");

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("expected object");
  return value as Record<string, unknown>;
}

function validate(payload: unknown) {
  const response = object(payload);
  if (response.model !== model) throw new Error(`unexpected model: ${String(response.model)}`);
  const usage = object(response.usage);
  if (typeof usage.input_tokens !== "number" || !Number.isFinite(usage.input_tokens) || usage.input_tokens < 0) {
    throw new Error("invalid input token usage");
  }
  const answers = object(response.answers);
  if (isBox) {
    for (const name of ["head", "absent"]) {
      const answer = object(answers[name]);
      const box = object(answer.box);
      for (const edge of ["left", "top", "right", "bottom"]) {
        if (typeof box[edge] !== "number" || !Number.isFinite(box[edge]) || box[edge] < 0 || box[edge] > 1) {
          throw new Error(`invalid box edge: ${edge}`);
        }
      }
      if (!(Number(box.left) < Number(box.right) && Number(box.top) < Number(box.bottom))) {
        throw new Error("invalid box order");
      }
    }
    for (const name of ["pet_present", "absent_present"]) {
      const value = object(answers[name]).noul;
      if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) throw new Error("invalid Noul");
    }
  } else if (!Object.hasOwn(criteria, String(object(answers.breed).choice))) {
    throw new Error("invalid breed choice");
  }
  return { model: response.model, answers, usage, cost_usd: usage.input_tokens * 0.042 / 1_000_000 };
}

async function ask(item: Item, withoutHeader = false) {
  if (isDecisions) {
    const photo = readFileSync(join(DIR, `data/${item.id}-384.jpg`));
    const instructions: DecisionQuestion["instructions"] = mode === "decisions-examples"
      ? [
        { type: "input_text", text: "Which of the six breeds is the main pet in the input photo? Reference photos follow." },
        ...manifest.items.filter((row) => row.split === "reference").flatMap((row) => [
          { type: "input_text" as const, text: `Reference for ${row.breed}:` },
          { type: "input_image" as const, image_url: `data:image/jpeg;base64,${readFileSync(join(DIR, `data/${row.id}-384.jpg`)).toString("base64")}` },
        ]),
      ]
      : "Which of the six breeds is the main pet in the input photo?";
    // Jev の state と同じく、入力は写真だけにする
    const result = await callDecisions([{ role: "user", content: [
      { type: "input_image", image_url: `data:image/jpeg;base64,${photo.toString("base64")}` },
    ] }], [{ type: "choice", name: "breed", instructions,
      choices: Object.entries(criteria).map(([value, description]) => ({ value, description })) }]);
    if (!result.ok) return result;
    const answer = result.answers.breed;
    if (answer?.type !== "choice" || !Object.hasOwn(criteria, answer.choice)) {
      return { ...result, ok: false, error: "unexpected breed answer" };
    }
    return { ...result, answers: { breed: answer } };
  }
  const assets: Record<string, { type: string; part: string }> = {};
  const uploads: { part: string; bytes: Buffer }[] = [];
  function image(row: Item, side: number) {
    const ref = randomUUID();
    const part = `image-${uploads.length}`;
    assets[ref] = { type: "image", part };
    uploads.push({ part, bytes: readFileSync(join(DIR, `data/${row.id}-${side}.jpg`)) });
    return ref;
  }
  const photo = image(item, isBox ? 784 : 384);
  const definitions: Record<string, unknown> = { ...criteria };
  if (mode.startsWith("examples-")) {
    for (const reference of manifest.items.filter((i) => i.split === "reference")) {
      definitions[reference.breed] = { description: criteria[reference.breed], example: image(reference, 384) };
    }
  }
  const questions = isBox ? {
    pet_present: { type: "noul", instructions: "Is a cat or dog visible in `photo`?" },
    head: { type: "box", instructions: BOX_INSTRUCTIONS },
    absent_present: { type: "noul", instructions: "Is a bicycle visible in `photo`?" },
    absent: { type: "box", instructions: "The bicycle in `photo`." },
  } : {
    breed: { type: "choice", instructions: BREED_INSTRUCTIONS, criteria: definitions },
  };
  const request = { model, state: { photo }, assets, questions };
  const form = new FormData();
  form.append("request", new Blob([JSON.stringify(request)], { type: "application/json" }));
  for (const upload of uploads) {
    form.append(upload.part, new Blob([new Uint8Array(upload.bytes)], { type: "image/jpeg" }), `${upload.part}.jpg`);
  }
  const started = performance.now();
  let status: number | undefined;
  try {
    const response = await fetch("https://api.typesafe.ai/v1/systemone", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, ...(withoutHeader ? {} : { "TypeSafe-Danger-Zone-Images": "true" }) },
      body: form, signal: AbortSignal.timeout(60_000),
    });
    status = response.status;
    const payload: unknown = await response.json();
    const duration_ms = performance.now() - started;
    if (!response.ok) return { ok: false, status, error: JSON.stringify(payload).slice(0, 1000), duration_ms };
    return { ok: true, status, ...validate(payload), duration_ms };
  } catch (error) {
    return { ok: false, status, error: error instanceof Error ? error.message : String(error), duration_ms: performance.now() - started };
  }
}

const settings = {
  version: 1, model: requestedModel, mode, split, concurrency: 2, retries: 0,
  classification_instructions: BREED_INSTRUCTIONS, box_instructions: BOX_INSTRUCTIONS,
  text_criteria: criteria, reference_ids: (!isDecisions || mode === "decisions-examples")
    ? manifest.items.filter((i) => i.split === "reference").map((i) => i.id) : [],
  images_header: !isDecisions && split !== "control", image_side: isBox ? 784 : 384,
  notes: isDecisions
    ? "Input is the photo only, as in Jev's state. No IDs, breed labels, filenames, or gold boxes are sent; images use base64 data URLs."
    : "No IDs, breed labels, filenames, or gold boxes are sent as state; multipart filenames are generic.",
};
const settingsPath = join(DIR, `settings-${mode}-${split}.json`);
if (existsSync(settingsPath) && readFileSync(settingsPath, "utf8") !== JSON.stringify(settings, null, 2) + "\n") {
  throw new Error("settings changed; use a new experiment version instead of mixing results");
}
writeFileSync(settingsPath, JSON.stringify(settings, null, 2) + "\n");
const outPath = join(DIR, `results-${mode}-${split}.jsonl`);
const previous = existsSync(outPath) ? readFileSync(outPath, "utf8").split("\n").filter(Boolean) : [];
const done = new Set(previous.map((line) => (JSON.parse(line) as { id: string }).id));
const targets = manifest.items.filter((i) => i.split === (split === "control" ? "dev" : split));
const selectedTargets = split === "control" ? targets.slice(0, 1) : targets;
const queue = selectedTargets.filter((i) => !done.has(i.id));
let failures = 0;
await Promise.all(Array.from({ length: 2 }, async () => {
  for (let item = queue.shift(); item !== undefined; item = queue.shift()) {
    const result = await ask(item, split === "control");
    if (!result.ok) failures++;
    const row = { id: item.id, split, mode, requested_model: requestedModel,
      gold: { breed: item.breed, species: item.species, head: item.box_784 },
      source_sha256: item.sha256, measured_at: new Date().toISOString(), result };
    appendFileSync(outPath, JSON.stringify(row) + "\n");
    console.log(`${item.id}: ${result.ok ? "ok" : "ERROR"}`);
  }
}));
console.log(`${outPath}: ${selectedTargets.length} targets, ${failures} new failures`);
if (failures > 0 && split !== "control") process.exitCode = 1;
