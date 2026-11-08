import { TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient(); // TYPESAFE_API_KEYを読む
const result = await client.systemOne({
  model: "jev-1.13.0",
  state: { message: "キャッシュカードがまだ届きません" },
  questions: {
    intent: {
      type: "choice",
      instructions: "銀行への問い合わせは、どの種類に当たるか。",
      criteria: {
        card_arrival: "キャッシュカードの配送状況の問い合わせ。",
        lost_card: "キャッシュカードをなくしたという連絡。",
        other: "それ以外の問い合わせ。",
      },
    },
    urgent: {
      type: "noul",
      instructions: "この問い合わせは、すぐに対応が要るか。",
    },
  },
});

const { choice, confidence } = result.answers.intent;
console.log("種類:", choice);
console.log("確信度:", confidence);
console.log("緊急である確率:", result.answers.urgent.noul);
console.log("入力トークン:", result.usage.input_tokens);
