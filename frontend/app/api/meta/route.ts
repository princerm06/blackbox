import { NextRequest, NextResponse } from "next/server";

const BASE_URL = process.env.META_BASE_URL ?? "https://api.meta.ai/v1";
const MODEL = process.env.META_MODEL ?? "muse-spark-1.3";

const HERO_MEMORY = {
  target: "My Keys",
  state: "unobserved",
  leading_association: { object: "My Bag", heuristic_weight: 0.58 },
  alternatives: [
    { hypothesis: "associated with person", heuristic_weight: 0.19 },
    { hypothesis: "last-seen area", heuristic_weight: 0.15 },
    { hypothesis: "outside observed scene", heuristic_weight: 0.07 },
  ],
  observed: [
    "My Keys were directly visible before becoming unobserved.",
    "My Bag was nearby around the disappearance window.",
    "My Bag moved after My Keys became unobserved.",
    "A person was present during part of the uncertainty window.",
  ],
  unknown: [
    "Whether the keys entered the bag.",
    "Whether a person physically moved the keys while occluded.",
    "The exact current physical location of the keys.",
  ],
  events: [
    { time_s: 20.3, event: "My Keys became unobserved" },
    { time_s: 23.4, event: "My Bag moved" },
    { time_s: 24.4, event: "person became unobserved" },
    { time_s: 31.9, event: "My Bag remained observed" },
  ],
};

function cleanJson(raw: string) {
  const trimmed = raw.trim();
  if (trimmed.startsWith("```")) {
    return trimmed.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  }
  return trimmed;
}

function messagesFor(question: string) {
  return [
    {
      role: "system",
      content:
        "You are the evidence-constrained investigation planner for BlackBox, a persistent visual-memory system. " +
        "Reason only from the supplied structured memory. Never convert spatial association into containment, transfer, manipulation, or causality unless directly observed. " +
        "Never call heuristic weights calibrated probabilities. Clearly separate observed facts, inference, and unknowns. " +
        "Return only valid JSON with keys answer, reasoning_summary, and next_checks. " +
        "next_checks must be an array of 1-3 objects with keys target, why, and expected_information_gain. Keep the answer concise.",
    },
    {
      role: "user",
      content: JSON.stringify({ question, memory: HERO_MEMORY }),
    },
  ];
}

export async function POST(request: NextRequest) {
  const apiKey = process.env.MODEL_API_KEY ?? process.env.META_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "MODEL_API_KEY is not configured on this deployment. Add your Meta Model API key to Vercel." },
      { status: 503 },
    );
  }

  const body = await request.json().catch(() => ({}));
  const question =
    typeof body.question === "string" && body.question.trim()
      ? body.question.trim()
      : "Where should I look for my keys next, and what evidence would reduce uncertainty most?";

  const response = await fetch(BASE_URL.replace(/\/$/, "") + "/chat/completions", {
    method: "POST",
    headers: {
      Authorization: "Bearer " + apiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      messages: messagesFor(question),
      temperature: 0.2,
      max_tokens: 1800,
    }),
    cache: "no-store",
  });

  if (!response.ok) {
    const detail = await response.text();
    return NextResponse.json(
      { error: "Meta Model API request failed", status: response.status, detail: detail.slice(0, 1200) },
      { status: 502 },
    );
  }

  const payload = await response.json();
  const raw = payload?.choices?.[0]?.message?.content;

  if (typeof raw !== "string" || !raw.trim()) {
    return NextResponse.json(
      { error: "Meta Model API returned no assistant content.", usage: payload?.usage ?? null },
      { status: 502 },
    );
  }

  const cleaned = cleanJson(raw);

  try {
    return NextResponse.json({
      provider: "Meta Model API",
      model: MODEL,
      source: "BlackBox structured hero-run memory",
      result: JSON.parse(cleaned),
    });
  } catch {
    return NextResponse.json({
      provider: "Meta Model API",
      model: MODEL,
      source: "BlackBox structured hero-run memory",
      result: {
        answer: cleaned,
        reasoning_summary: "Muse Spark returned text rather than the requested JSON structure.",
        next_checks: [],
      },
    });
  }
}
