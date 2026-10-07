"use client";

import { FormEvent, useMemo, useState } from "react";

import {
  API_BASE,
  RunSnapshot,
  TrackingTarget,
  createSession,
  getRun,
  mediaUrl,
  registerTarget,
  removeTarget,
  setTargetEnabled,
  uploadVideo,
  whereIs,
} from "../lib/api";

type QueryResult = Awaited<ReturnType<typeof whereIs>>;

export default function HomePage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [targets, setTargets] = useState<TrackingTarget[]>([]);
  const [displayName, setDisplayName] = useState("");
  const [prompt, setPrompt] = useState("");
  const [video, setVideo] = useState<File | null>(null);
  const [run, setRun] = useState<RunSnapshot | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [queryTargetId, setQueryTargetId] = useState("");
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);

  const selectedTarget = useMemo(
    () => targets.find((target) => target.target_id === queryTargetId),
    [targets, queryTargetId],
  );

  async function ensureSession() {
    if (sessionId) return sessionId;
    setStatus("Starting session…");
    const created = await createSession();
    setSessionId(created.session_id);
    setStatus("Session ready");
    return created.session_id;
  }

  async function handleAddTarget(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const activeSession = await ensureSession();
      const created = await registerTarget(
        activeSession,
        displayName.trim(),
        prompt.trim() || displayName.trim(),
      );
      setTargets((current) => [...current, created]);
      setQueryTargetId((current) => current || created.target_id);
      setDisplayName("");
      setPrompt("");
      setStatus("Target added");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not add target");
    }
  }

  async function handleToggleTarget(target: TrackingTarget) {
    if (!sessionId) return;
    setError(null);
    try {
      const updated = await setTargetEnabled(
        sessionId,
        target.target_id,
        !target.enabled,
      );
      setTargets((current) =>
        current.map((item) =>
          item.target_id === updated.target_id ? updated : item,
        ),
      );
      if (!updated.enabled && queryTargetId === updated.target_id) {
        setQueryTargetId("");
      }
      setStatus(updated.enabled ? "Target enabled" : "Target paused");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update target");
    }
  }

  async function handleRemoveTarget(target: TrackingTarget) {
    if (!sessionId) return;
    setError(null);
    try {
      await removeTarget(sessionId, target.target_id);
      setTargets((current) =>
        current.filter((item) => item.target_id !== target.target_id),
      );
      if (queryTargetId === target.target_id) {
        setQueryTargetId("");
        setQueryResult(null);
      }
      setStatus("Target removed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not remove target");
    }
  }

  async function handleProcess() {
    if (!video) {
      setError("Choose a video first.");
      return;
    }
    setError(null);
    setQueryResult(null);
    try {
      const activeSession = await ensureSession();
      setStatus("Processing video…");
      const uploaded = await uploadVideo(activeSession, video);
      const snapshot = await getRun(activeSession, uploaded.run_id);
      setRun(snapshot);
      setStatus("Run ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Processing failed");
      setStatus("Ready");
    }
  }

  async function handleWhereQuery() {
    if (!sessionId || !run || !queryTargetId) {
      setError("Process a run and choose a tracked object first.");
      return;
    }
    setError(null);
    setStatus("Checking memory…");
    try {
      const result = await whereIs(sessionId, run.run_id, queryTargetId);
      setQueryResult(result);
      setStatus("Answer ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
      setStatus("Run ready");
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">BLACKBOX / VISUAL MEMORY</p>
          <h1>Remember what the camera saw.</h1>
        </div>
        <div className="status">
          <span className="statusDot" />
          {status}
        </div>
      </header>

      {error && <div className="errorBanner">{error}</div>}

      <section className="workspace">
        <div className="primaryColumn">
          <section className="panel videoPanel">
            <div className="panelHeader">
              <div>
                <p className="panelLabel">Observation</p>
                <h2>{run ? "Processed scene" : "Add a scene"}</h2>
              </div>
              {run && (
                <span className="muted">
                  {run.objects.length} object{run.objects.length === 1 ? "" : "s"}
                </span>
              )}
            </div>

            {run?.media.annotated ? (
              <video className="video" controls src={mediaUrl(run.media.annotated)} />
            ) : (
              <label className="dropzone">
                <input
                  type="file"
                  accept="video/*"
                  onChange={(event) => setVideo(event.target.files?.[0] ?? null)}
                />
                <span className="dropIcon">＋</span>
                <strong>{video ? video.name : "Choose a video"}</strong>
                <span>A fixed-camera clip works best for the current prototype.</span>
              </label>
            )}

            <div className="videoActions">
              <button
                className="button buttonPrimary"
                onClick={handleProcess}
                disabled={!video || targets.filter((target) => target.enabled).length === 0}
              >
                Process scene
              </button>
              <span className="muted">
                {targets.filter((target) => target.enabled).length === 0
                  ? "Add or enable at least one tracked object."
                  : targets.filter((target) => target.enabled).length + " target" + (targets.filter((target) => target.enabled).length === 1 ? "" : "s") + " active"}
              </span>
            </div>
          </section>

          <section className="panel memoryPanel">
            <div className="panelHeader">
              <div>
                <p className="panelLabel">Memory</p>
                <h2>Ask BlackBox</h2>
              </div>
            </div>

            <div className="queryRow">
              <select
                value={queryTargetId}
                onChange={(event) => setQueryTargetId(event.target.value)}
              >
                <option value="">Choose an object</option>
                {targets.filter((target) => target.enabled).map((target) => (
                  <option key={target.target_id} value={target.target_id}>
                    {target.display_name}
                  </option>
                ))}
              </select>
              <button
                className="button"
                onClick={handleWhereQuery}
                disabled={!run || !queryTargetId}
              >
                Where is it?
              </button>
            </div>

            {queryResult ? (
              <div className="answerCard">
                <div className="answerMeta">
                  <span className={"provenance provenance-" + queryResult.answer.provenance}>
                    {queryResult.answer.provenance}
                  </span>
                  <strong>{Math.round(queryResult.answer.confidence * 100)}%</strong>
                </div>
                <p className="answerText">{queryResult.answer.answer}</p>

                {queryResult.answer.evidence.length > 0 && (
                  <div className="detailBlock">
                    <span className="detailLabel">Evidence</span>
                    {queryResult.answer.evidence.map((item) => (
                      <p key={item}>{item}</p>
                    ))}
                  </div>
                )}

                {queryResult.reobservation.should_reobserve && (
                  <div className="reobserve">
                    <span className="detailLabel">Next check</span>
                    <p>{queryResult.reobservation.rationale}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="emptyAnswer">
                {selectedTarget
                  ? "Ask about " + selectedTarget.display_name + " after processing a scene."
                  : "Select a tracked object to query its visual memory."}
              </div>
            )}
          </section>
        </div>

        <aside className="sideColumn">
          <section className="panel">
            <div className="panelHeader compact">
              <div>
                <p className="panelLabel">Targets</p>
                <h2>Tracked objects</h2>
              </div>
            </div>

            <form className="targetForm" onSubmit={handleAddTarget}>
              <input
                required
                placeholder="My Keys"
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
              />
              <input
                placeholder="visual prompt: keys"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
              />
              <button className="button" type="submit">Add object</button>
            </form>

            <div className="targetList">
              {targets.length === 0 ? (
                <p className="muted">No tracked objects yet.</p>
              ) : (
                targets.map((target) => (
                  <div className="targetRow" key={target.target_id}>
                    <div className="targetInfo">
                      <strong>{target.display_name}</strong>
                      <span>{target.prompt}</span>
                    </div>
                    <div className="targetActions">
                      <button
                        type="button"
                        className="targetAction"
                        onClick={() => handleToggleTarget(target)}
                      >
                        {target.enabled ? "Pause" : "Resume"}
                      </button>
                      <button
                        type="button"
                        className="targetAction dangerAction"
                        onClick={() => handleRemoveTarget(target)}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panelHeader compact">
              <div>
                <p className="panelLabel">State</p>
                <h2>Objects</h2>
              </div>
            </div>

            <div className="objectList">
              {!run || run.objects.length === 0 ? (
                <p className="muted">Process a scene to populate memory.</p>
              ) : (
                run.objects.map((object) => (
                  <div className="objectRow" key={object.object_id}>
                    <div>
                      <strong>{object.class_name ?? "unknown"}</strong>
                      <span>
                        {object.visibility === "observed"
                          ? "currently observed"
                          : "not currently observed"}
                      </span>
                    </div>
                    <span className={object.has_uncertainty ? "uncertain" : "stable"}>
                      {object.has_uncertainty ? "uncertain" : "grounded"}
                    </span>
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="panel timelinePanel">
            <div className="panelHeader compact">
              <div>
                <p className="panelLabel">Timeline</p>
                <h2>What changed</h2>
              </div>
            </div>

            <div className="timeline">
              {!run || run.events.length === 0 ? (
                <p className="muted">No events yet.</p>
              ) : (
                [...run.events].reverse().slice(0, 8).map((event, index) => (
                  <div
                    className="timelineItem"
                    key={event.object_id + "-" + event.frame_index + "-" + index}
                  >
                    <span className="timelineDot" />
                    <div>
                      <strong>
                        {event.event_type.replaceAll("_", " ").toLowerCase()}
                      </strong>
                      <span>
                        {event.class_name} · {(event.timestamp_ms / 1000).toFixed(1)}s · {event.provenance}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>
      </section>

      <footer>API: {API_BASE}</footer>
    </main>
  );
}
