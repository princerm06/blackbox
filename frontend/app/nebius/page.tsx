"use client";

import { useState } from "react";
import styles from "./nebius.module.css";

type NextCheck = {
  target?: string;
  why?: string;
  expected_information_gain?: string;
};

type Result = {
  answer?: string;
  reasoning_summary?: string;
  next_checks?: NextCheck[];
};

type ApiResponse = {
  provider?: string;
  model?: string;
  source?: string;
  result?: Result;
  error?: string;
  detail?: string;
};

const questions = [
  "Where should I look for my keys next, and why?",
  "What can you infer without claiming the keys are inside the bag?",
  "What evidence would reduce uncertainty the most?",
];

export default function NebiusDemoPage() {
  const [question, setQuestion] = useState(questions[0]);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function investigate() {
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch("/api/nebius", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const payload = (await response.json()) as ApiResponse;
      setResult(payload);
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : "Request failed" });
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <div className={styles.kicker}>BLACKBOX × NEBIUS × NVIDIA</div>
          <h1>Visual memory that plans what to inspect next.</h1>
          <p>
            BlackBox turns video into structured world state. NVIDIA Nemotron 3.5
            Lightning, served through Nebius Token Factory, reasons over that
            evidence to produce grounded next-check plans without upgrading
            association into unsupported certainty.
          </p>
        </div>
        <a className={styles.back} href="/">Open full BlackBox →</a>
      </header>

      <section className={styles.grid}>
        <aside className={styles.memory}>
          <div className={styles.label}>BLACKBOX MEMORY SNAPSHOT</div>
          <h2>My Keys</h2>
          <div className={styles.status}>UNOBSERVED · leading association: My Bag · 58% heuristic weight</div>

          <div className={styles.block}>
            <strong>Observed</strong>
            <p>✓ Keys were visible before becoming unobserved.</p>
            <p>✓ Bag was nearby around the disappearance window.</p>
            <p>✓ Bag moved afterward.</p>
            <p>✓ A person was present during part of the uncertainty window.</p>
          </div>

          <div className={styles.block}>
            <strong>Still unknown</strong>
            <p>? Whether the keys entered the bag.</p>
            <p>? Whether a person physically moved the keys while occluded.</p>
            <p>? Exact current physical location.</p>
          </div>

          <div className={styles.timeline}>
            <span>20.3s · keys unobserved</span>
            <span>23.4s · bag moved</span>
            <span>24.4s · person unobserved</span>
            <span>31.9s · bag observed</span>
          </div>
        </aside>

        <section className={styles.agent}>
          <div className={styles.badges}>
            <span>Nebius Token Factory</span>
            <span>NVIDIA Nemotron 3.5 Lightning</span>
            <span>Live runtime inference</span>
          </div>

          <div className={styles.label}>ASK THE MEMORY INVESTIGATOR</div>
          <div className={styles.prompts}>
            {questions.map((item) => (
              <button
                key={item}
                className={item === question ? styles.active : styles.prompt}
                onClick={() => setQuestion(item)}
              >
                {item}
              </button>
            ))}
          </div>

          <div className={styles.query}>{question}</div>
          <button className={styles.run} disabled={loading} onClick={investigate}>
            {loading ? "Calling Nemotron…" : "Run Nemotron investigation"}
          </button>

          {result?.error && (
            <div className={styles.error}>
              <strong>Configuration / API error</strong>
              <p>{result.error}</p>
              {result.detail && <code>{result.detail}</code>}
            </div>
          )}

          {result?.result && (
            <div className={styles.output}>
              <div className={styles.modelLine}>
                {result.provider} · {result.model}
              </div>
              <h2>Grounded answer</h2>
              <p className={styles.answer}>{result.result.answer}</p>

              <h3>Reasoning summary</h3>
              <p>{result.result.reasoning_summary}</p>

              <h3>Next checks</h3>
              <div className={styles.checks}>
                {(result.result.next_checks ?? []).map((check, index) => (
                  <div key={`${check.target ?? "check"}-${index}`} className={styles.check}>
                    <strong>{check.target ?? `Check ${index + 1}`}</strong>
                    <p>{check.why}</p>
                    <span>{check.expected_information_gain}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className={styles.disclosure}>
            The hosted challenge view uses a preprocessed BlackBox hero-run memory
            snapshot. The reasoning request itself is live and is sent at runtime
            to an NVIDIA Nemotron model through Nebius Token Factory.
          </div>
        </section>
      </section>
    </main>
  );
}
