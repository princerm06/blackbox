"use client";

import { useState } from "react";
import styles from "./amazon.module.css";

type DemoResponse = {
  tool: string;
  title: string;
  answer: string;
  observed?: string[];
  inferred?: string[];
  unknown?: string[];
  evidence?: string[];
  next?: string;
};

const RESPONSES: Record<string, DemoResponse> = {
  "Where are my keys?": {
    tool: "where_are_my_keys_tool",
    title: "Location memory",
    answer:
      "The leading hypothesis is that My Keys is associated with My Bag (58% heuristic weight).",
    evidence: [
      "My Keys were directly observed before becoming unobserved.",
      "My Bag was nearby around the disappearance window.",
      "My Bag moved after the keys became unobserved.",
      "A person was present during part of the uncertainty window.",
    ],
    next: "Inspect My Bag first to reduce uncertainty.",
  },
  "What changed?": {
    tool: "what_changed_tool",
    title: "Temporal memory",
    answer:
      "The keys became unobserved, the bag moved afterward, and the person later left the observed scene.",
    evidence: [
      "20.3s — My Keys became unobserved",
      "23.4s — My Bag moved",
      "24.4s — person became unobserved",
      "31.9s — My Bag remained observed",
    ],
  },
  "What do you actually know?": {
    tool: "what_do_you_know_tool",
    title: "Provenance check",
    answer:
      "BlackBox separates what the camera directly observed from what the system inferred and what remains unknown.",
    observed: [
      "My Keys were visible before the disappearance event.",
      "My Bag was spatially associated with the keys around that time.",
      "My Bag moved after My Keys became unobserved.",
    ],
    inferred: ["My Bag is the leading association for the missing keys."],
    unknown: [
      "Whether the keys entered the bag.",
      "Whether the person moved the keys while they were occluded.",
      "The keys' exact current physical location.",
    ],
  },
  "What should I check next?": {
    tool: "recommended_check_tool",
    title: "Re-observation plan",
    answer:
      "Check My Bag first. It is the leading associated object and inspecting it should reduce uncertainty most efficiently.",
    next: "Recommended target: My Bag · re-observation score 82%",
  },
};

const prompts = Object.keys(RESPONSES);

export default function AmazonDemoPage() {
  const [question, setQuestion] = useState(prompts[0]);
  const [history, setHistory] = useState<string[]>([]);
  const response = RESPONSES[question];

  function ask(prompt: string) {
    setQuestion(prompt);
    setHistory((current) => [prompt, ...current.filter((item) => item !== prompt)].slice(0, 4));
  }

  return (
    <main className={styles.page}>
      <div className={styles.glowA} />
      <div className={styles.glowB} />

      <header className={styles.header}>
        <div>
          <div className={styles.kicker}>BLACKBOX × ALEXA+ · AMAZON BUILD, SHIP, SHAPE</div>
          <h1>Ask your camera what it remembers.</h1>
          <p>
            A conversational layer over BlackBox&apos;s persistent visual memory.
            The Alexa+ simulation maps user requests to a self-hosted MCP server,
            while BlackBox keeps observed facts separate from inference.
          </p>
        </div>
        <div className={styles.headerActions}>
          <div className={styles.badges}>
            <span>Alexa+ simulation</span>
            <span>MCP · Streamable HTTP</span>
            <span>Persistent context</span>
          </div>
          <a className={styles.fullBlackBoxLink} href="/">
            Open full BlackBox →
          </a>
        </div>
      </header>

      <section className={styles.grid}>
        <aside className={styles.sceneCard}>
          <div className={styles.cardTop}>
            <span>Remembered scene</span>
            <span className={styles.live}>MEMORY READY</span>
          </div>

          <div className={styles.scene}>
            <div className={styles.object + " " + styles.keys}>
              <div className={styles.objectIcon}>⌁</div>
              <strong>My Keys</strong>
              <span>UNOBSERVED</span>
            </div>
            <div className={styles.path} />
            <div className={styles.object + " " + styles.bag}>
              <div className={styles.objectIcon}>▣</div>
              <strong>My Bag</strong>
              <span>OBSERVED</span>
            </div>
            <div className={styles.person}>PERSON<br /><span>UNOBSERVED</span></div>
          </div>

          <div className={styles.timeline}>
            <div><b>20.3s</b><span>Keys became unobserved</span></div>
            <div><b>23.4s</b><span>Bag moved</span></div>
            <div><b>24.4s</b><span>Person became unobserved</span></div>
          </div>

          <div className={styles.disclosure}>
            Hosted hackathon demo uses a preprocessed BlackBox hero run generated
            by the full local perception pipeline.
          </div>
        </aside>

        <section className={styles.conversation}>
          <div className={styles.alexaOrb}>
            <div className={styles.orbInner} />
          </div>

          <div className={styles.promptLabel}>TRY AN ALEXA+ REQUEST</div>
          <div className={styles.promptRow}>
            {prompts.map((prompt) => (
              <button
                key={prompt}
                onClick={() => ask(prompt)}
                className={question === prompt ? styles.activePrompt : styles.prompt}
              >
                “{prompt}”
              </button>
            ))}
          </div>

          <div className={styles.transcript}>
            <div className={styles.userLine}>
              <span>YOU</span>
              <p>Alexa, {question.charAt(0).toLowerCase() + question.slice(1)}</p>
            </div>

            <div className={styles.toolLine}>
              <span>MCP TOOL</span>
              <code>{response.tool}</code>
            </div>

            <div className={styles.assistantLine}>
              <span>BLACKBOX MEMORY</span>
              <h2>{response.title}</h2>
              <p className={styles.answer}>{response.answer}</p>

              {response.evidence && (
                <div className={styles.section}>
                  <strong>Evidence</strong>
                  {response.evidence.map((item) => <p key={item}>✓ {item}</p>)}
                </div>
              )}

              {response.observed && (
                <div className={styles.threeCols}>
                  <div>
                    <strong>Observed</strong>
                    {response.observed.map((item) => <p key={item}>✓ {item}</p>)}
                  </div>
                  <div>
                    <strong>Inferred</strong>
                    {response.inferred?.map((item) => <p key={item}>~ {item}</p>)}
                  </div>
                  <div>
                    <strong>Unknown</strong>
                    {response.unknown?.map((item) => <p key={item}>? {item}</p>)}
                  </div>
                </div>
              )}

              {response.next && <div className={styles.next}>{response.next}</div>}
            </div>
          </div>

          <div className={styles.footerLine}>
            <span>Persistent memory across requests</span>
            <span>Observed ≠ inferred</span>
            <span>No containment claim without evidence</span>
          </div>
        </section>
      </section>
    </main>
  );
}
