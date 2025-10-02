export type A01In = { prompt: string; criteria?: string[]; };
export type A01Out = { answer: string; critic?: string; };

export type RAGIn = { question: string; k?: number; };
export type RAGOut = { answer: string; sources: string[]; };

export type A02In = { question: string; };
export type A02Out = { answer: string; sources: string[]; };

const API_BASE = ""; // same origin

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${txt}`);
  }
  return res.json() as Promise<T>;
}

export const API = {
  health: () => jsonFetch<{ ok: boolean; env: Record<string, string>; }>(`${API_BASE}/healthz`),
  a01: (body: A01In) =>
    jsonFetch<A01Out>(`${API_BASE}/api/a01/run`, { method: "POST", body: JSON.stringify(body) }),
  rag: (body: RAGIn) =>
    jsonFetch<RAGOut>(`${API_BASE}/api/rag/qa`, { method: "POST", body: JSON.stringify(body) }),
  a02: (body: A02In) =>
    jsonFetch<A02Out>(`${API_BASE}/api/a02/run`, { method: "POST", body: JSON.stringify(body) }),
};
