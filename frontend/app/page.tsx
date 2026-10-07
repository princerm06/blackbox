"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  API_BASE,
  RunSnapshot,
  TrackingTarget,
  createSession,
  getRun,
  listTargets,
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
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("Ready");
  const [mediaMode, setMediaMode] = useState<"annotated" | "source">("annotated");
  const [error, setError] = useState<string | null>(null);
  const [selectedEventKey, setSelectedEventKey] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const saved = window.localStorage.getItem("blackbox.session_id");
    if (!saved) return;

    setStatus("Restoring session…");
    listTargets(saved)
      .then((payload) => {
        setSessionId(saved);
        setTargets(payload.targets);
        const firstEnabled = payload.targets.find((target) => target.enabled);
        if (firstEnabled) {
          setQueryTargetId(firstEnabled.target_id);
        }
        setStatus("Session restored");
      })
      .catch(() => {
        window.localStorage.removeItem("blackbox.session_id");
        setStatus("Ready");
      });
  }, []);

  const selectedTarget = useMemo(
    () => targets.find((target) => target.target_id === queryTargetId),
    [targets, queryTargetId],
  );

  async function ensureSession() {
    if (sessionId) return sessionId;
    setStatus("Starting session…");
    const created = await createSession();
    setSessionId(created.session_id);
    window.localStorage.setItem("blackbox.session_id", created.session_id);
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

  function seekTo(timestampMs: number, eventKey?: string) {
    const player = videoRef.current;
    if (!player) return;

    player.currentTime = Math.max(0, timestampMs / 1000);
    player.play().catch(() => undefined);
    if (eventKey) {
      setSelectedEventKey(eventKey);
    }
  }

  function handleNewScene() {
    setRun(null);
    setVideo(null);
    setQueryResult(null);
    setQuestion("");
    setMediaMode("annotated");
    setError(null);
    setStatus("Ready for another scene");
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

  function resolveQuestionTarget() {
    const enabledTargets = targets.filter((target) => target.enabled);
    if (queryTargetId) {
      const selected = enabledTargets.find(
        (target) => target.target_id === queryTargetId,
      );
      if (selected) return selected;
    }

    const normalized = question.toLowerCase();
    const mentioned = enabledTargets.find((target) => {
      const displayName = target.display_name.toLowerCase();
      const promptName = target.prompt.toLowerCase();
      return (
        normalized.includes(displayName) ||
        normalized.includes(promptName)
      );
    });

    if (mentioned) return mentioned;
    if (enabledTargets.length === 1) return enabledTargets[0];
    return null;
  }

  async function handleWhereQuery(event?: FormEvent) {
    event?.preventDefault();
    const target = resolveQuestionTarget();

    if (!sessionId || !run || !target) {
      setError(
        "Process a run, then choose or mention one tracked object.",
      );
      return;
    }

    if (
      question.trim() &&
      !/where|last seen|location|find/i.test(question)
    ) {
      setError(
        "The current demo supports location and last-seen questions first.",
      );
      return;
    }

    setQueryTargetId(target.target_id);
    setError(null);
    setStatus("Checking memory…");
    try {
      const result = await whereIs(sessionId, run.run_id, target.target_id);
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

            {run ? (
              <>
                <div className="mediaSwitch">
                  <button
                    className={mediaMode === "annotated" ? "mediaTab activeMediaTab" : "mediaTab"}
                    onClick={() => setMediaMode("annotated")}
                    disabled={!run.media.annotated}
                  >
                    BlackBox view
                  </button>
                  <button
                    className={mediaMode === "source" ? "mediaTab activeMediaTab" : "mediaTab"}
                    onClick={() => setMediaMode("source")}
                  >
                    Source
                  </button>
                </div>
                <video
                  ref={videoRef}
                  className="video"
                  controls
                  src={mediaUrl(
                    mediaMode === "annotated" && run.media.annotated
                      ? run.media.annotated
                      : run.media.source,
                  )}
                />
              </>
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
              {run ? (
                <button className="button" onClick={handleNewScene}>
                  New scene
                </button>
              ) : (
                <button
                  className="button buttonPrimary"
                  onClick={handleProcess}
                  disabled={!video || targets.filter((target) => target.enabled).length === 0}
                >
                  Process scene
                </button>
              )}
              <span className="muted">
                {targets.filter((target) => target.enabled).length === 0
                  ? "Add or enable at least one tracked object."
                  : targets.filter((target) => target.enabled).length + " target" + (targets.filter((target) => target.enabled).length === 1 ? "" : "s") + " active"}
              </span>
            </div>

            {run && (
              <div className="runStats">
                <div>
                  <span>Frames</span>
                  <strong>{run.summary.frames_processed}</strong>
                </div>
                <div>
                  <span>Detections</span>
                  <strong>{run.summary.detections_emitted}</strong>
                </div>
                <div>
                  <span>Tracks</span>
                  <strong>{run.summary.track_observations_emitted}</strong>
                </div>
                <div>
                  <span>Events</span>
                  <strong>{run.summary.events_emitted}</strong>
                </div>
              </div>
            )}
          </section>

          <section className="panel memoryPanel">
            <div className="panelHeader">
              <div>
                <p className="panelLabel">Memory</p>
                <h2>Ask BlackBox</h2>
              </div>
            </div>

            <form className="queryComposer" onSubmit={handleWhereQuery}>
              <input
                className="questionInput"
                placeholder="Where are my keys?"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
              />
              <button
                className="button"
                type="submit"
                disabled={!run || targets.filter((target) => target.enabled).length === 0}
              >
                Ask
              </button>
            </form>

            <div className="queryTargets">
              {targets.filter((target) => target.enabled).map((target) => (
                <button
                  key={target.target_id}
                  type="button"
                  className={
                    queryTargetId === target.target_id
                      ? "queryTarget activeQueryTarget"
                      : "queryTarget"
                  }
                  onClick={() => setQueryTargetId(target.target_id)}
                >
                  {target.display_name}
                </button>
              ))}
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

                {queryResult.ambiguity && (
                  <div className="ambiguityNotice">
                    {queryResult.ambiguity}
                  </div>
                )}

                {queryResult.answer.alternatives.length > 0 && (
                  <div className="detailBlock">
                    <span className="detailLabel">Alternatives</span>
                    {queryResult.answer.alternatives.map((item) => (
                      <div className="hypothesisItem" key={item.hypothesis}>
                        <div className="hypothesisRow">
                          <span>{item.hypothesis.replaceAll("_", " ")}</span>
                          <strong>{Math.round(item.weight * 100)}%</strong>
                        </div>
                        <div className="confidenceTrack">
                          <div
                            className="confidenceFill"
                            style={{ width: Math.round(item.weight * 100) + "%" }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {queryResult.answer.evidence.length > 0 && (
                  <div className="detailBlock">
                    <span className="detailLabel">Evidence</span>
                    {queryResult.answer.evidence.map((item) => (
                      <p key={item}>{item}</p>
                    ))}
                  </div>
                )}

                {queryResult.answer.caveats.length > 0 && (
                  <div className="detailBlock">
                    <span className="detailLabel">Caveats</span>
                    {queryResult.answer.caveats.map((item) => (
                      <p key={item}>{item}</p>
                    ))}
                  </div>
                )}

                {queryResult.reobservation.should_reobserve && (
                  <div className="reobserve">
                    <span className="detailLabel">Next check</span>
                    <p>{queryResult.reobservation.rationale}</p>
                    {queryResult.reobservation.targets.map((target) => (
                      <div className="hypothesisRow" key={target.target}>
                        <span>{target.target.replaceAll(":", " → ")}</span>
                        <strong>{Math.round(target.score * 100)}%</strong>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="emptyAnswer">
                {selectedTarget
                  ? "Ask a location question about " + selectedTarget.display_name + "."
                  : "Ask where a tracked object is, or select one above."}
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
                    <div className="objectInfo">
                      <strong>{object.class_name ?? "unknown"}</strong>
                      <span>
                        {object.visibility === "observed"
                          ? "currently observed"
                          : "not currently observed"}
                      </span>
                      <div className="confidenceTrack objectConfidence">
                        <div
                          className="confidenceFill"
                          style={{ width: Math.round(object.visibility_confidence * 100) + "%" }}
                        />
                      </div>
                    </div>
                    <div className="objectActions">
                      {object.last_observed_timestamp_ms !== null && (
                        <button
                          type="button"
                          className="targetAction"
                          onClick={() => seekTo(object.last_observed_timestamp_ms ?? 0)}
                        >
                          Last seen
                        </button>
                      )}
                      <span className={object.has_uncertainty ? "uncertain" : "stable"}>
                        {object.has_uncertainty ? "uncertain" : "grounded"}
                      </span>
                    </div>
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
                [...run.events].reverse().slice(0, 8).map((event, index) => {
                  const eventKey =
                    event.object_id + "-" + event.frame_index + "-" + index;
                  return (
                    <button
                      type="button"
                      className={
                        selectedEventKey === eventKey
                          ? "timelineItem selectedTimelineItem"
                          : "timelineItem"
                      }
                      key={eventKey}
                      onClick={() => seekTo(event.timestamp_ms, eventKey)}
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
                      <span className="timelineJump">View</span>
                    </button>
                  );
                })
              )}
            </div>
          </section>
        </aside>
      </section>

      <footer>API: {API_BASE}</footer>
    </main>
  );
}
