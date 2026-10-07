export const API_BASE =
  process.env.NEXT_PUBLIC_BLACKBOX_API ?? "http://127.0.0.1:8000";

export type TrackingTarget = {
  target_id: string;
  display_name: string;
  prompt: string;
  enabled: boolean;
};

export type ObjectSummary = {
  object_id: string;
  class_name: string | null;
  visibility: string;
  visibility_confidence: number;
  last_observed_frame: number | null;
  last_observed_timestamp_ms: number | null;
  has_uncertainty: boolean;
};

export type TimelineEvent = {
  object_id: string;
  event_type: string;
  frame_index: number;
  timestamp_ms: number;
  class_name: string;
  confidence: number;
  provenance: string;
};

export type RunSnapshot = {
  session_id: string;
  run_id: string;
  objects: ObjectSummary[];
  events: TimelineEvent[];
  media: {
    source: string;
    annotated: string | null;
  };
};

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `Request failed: ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export function createSession() {
  return request<{ session_id: string }>("/sessions", {
    method: "POST",
  });
}

export function registerTarget(
  sessionId: string,
  displayName: string,
  prompt: string,
) {
  return request<TrackingTarget>(
    `/sessions/${sessionId}/targets`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        display_name: displayName,
        prompt,
      }),
    },
  );
}

export function listTargets(sessionId: string) {
  return request<{ targets: TrackingTarget[] }>(
    `/sessions/${sessionId}/targets`,
  );
}

export async function uploadVideo(
  sessionId: string,
  file: File,
) {
  const form = new FormData();
  form.append("video", file);

  return request<{ run_id: string }>(
    `/sessions/${sessionId}/videos`,
    {
      method: "POST",
      body: form,
    },
  );
}

export function getRun(sessionId: string, runId: string) {
  return request<RunSnapshot>(
    `/sessions/${sessionId}/runs/${runId}`,
  );
}

export function whereIs(
  sessionId: string,
  runId: string,
  targetId: string,
) {
  return request<{
    answer: {
      answer: string;
      confidence: number;
      provenance: string;
      alternatives: Array<{
        hypothesis: string;
        weight: number;
      }>;
      evidence: string[];
      caveats: string[];
    };
    reobservation: {
      should_reobserve: boolean;
      uncertainty: number;
      rationale: string;
      targets: Array<{
        target: string;
        score: number;
        rationale: string;
      }>;
    };
    ambiguity: string | null;
  }>(
    `/sessions/${sessionId}/runs/${runId}/targets/${targetId}/where`,
  );
}

export function mediaUrl(path: string) {
  return `${API_BASE}${path}`;
}
