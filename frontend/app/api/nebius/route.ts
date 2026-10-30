import { NextRequest, NextResponse } from "next/server";

const BASE_URL =
  process.env.NEBIUS_BASE_URL ?? "https://api.tokenfactory.nebius.com/v1";
const MODEL =
  process.env.NEBIUS_MODEL ?? "nvidia/Nemotron-3_5-Lightning";

const HERO_MEMORY = {
  target: "My Keys",
  state: "unobserved",
  leading_association: {
    object: "My Bag",
    heuristic_weight: 0.58,
  },
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

function promptFor(question: string) {
  return [
    {
      role: "system",
      content:
        "You are the evidence-constrained investigation planner for BlackBox, a persistent visual-memory system. " +
        "Reason only from the supplied structured memory. Never convert spatial association into containment, transfer, manipulation, or causality unless directly observed. " +
        "Never call heuristic weights calibrated probabilities. Clearly separate observed facts, inference, and unknowns. " +
        "Return ONLY valid JSON with keys: answer, reasoning_summary, next_checks. " +
        "next_checks must be an array of 1-3 objects with keys target, why, expected_information_gain. Keep the answer concise.",
    },
    {
      role: "user",
      content: JSON.stringify({
        question,
        memory: HERO_MEMORY,
      }),
    },
  ];
}

export async function POST(request: NextRequest) {
  const apiKey = process.env.NEBIUS_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      {
        error:
          "NEBIUS_API_KEY is not configured on this deployment. Add it to the Vercel environment to enable live Token Factory inference.",
      },
      { status: 503 },
    );
  }

  const body = await request.json().catch(() => ({}));
  const question =
    typeof body.question === "string" && body.question.trim()
      ? body.question.trim()
      : "Where should I look for my keys next, and why?";

  const response = await fetch(`${BASE_URL.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      messages: promptFor(question),
      temperature: 0.2,
      max_tokens: 6000,
    }),
    cache: "no-store",
  });

  if (!response.ok) {
    const detail = await response.text();
    return NextResponse.json(
      {
        error: "Nebius Token Factory request failed",
        status: response.status,
        detail: detail.slice(0, 1200),
      },
      { status: 502 },
    );
  }

  const payload = await response.json();
  const choice = payload?.choices?.[0];
  const raw = choice?.message?.content;

  if (typeof raw !== "string" || !raw.trim()) {
    return NextResponse.json(
      {
        error: "Token Factory returned no visible assistant answer.",
        detail:
          choice?.finish_reason === "length"
            ? "Nemotron exhausted its output budget while reasoning. Retry the request."
            : `finish_reason=${choice?.finish_reason ?? "unknown"}`,
      },
      { status: 502 },
    );
  }

  const cleaned = raw
    .trim()
    .replace(/^\`\`\`(?:json)?\s*/i, "")
    .replace(/\s*\`\`\`$/, "");

  try {
    const parsed = JSON.parse(cleaned);
    return NextResponse.json({
      provider: "Nebius Token Factory",
      model: MODEL,
      source: "BlackBox structured hero-run memory",
      result: parsed,
    });
  } catch {
    return NextResponse.json({
      provider: "Nebius Token Factory",
      model: MODEL,
      source: "BlackBox structured hero-run memory",
      result: {
        answer: cleaned,
        reasoning_summary:
          "Nemotron returned text instead of the requested JSON structure.",
        next_checks: [],
      },
    });
  }
}
