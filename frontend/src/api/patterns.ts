import type { PatternMeta } from "../types/patterns";

export async function fetchPatterns(): Promise<PatternMeta[]> {
  const r = await fetch("/api/patterns");
  if (!r.ok) throw new Error("fetch patterns failed");
  const j = await r.json();
  return j.items as PatternMeta[];
}

export async function fetchPattern(id: string): Promise<PatternMeta> {
  const r = await fetch(`/api/patterns/${id}`);
  if (!r.ok) throw new Error(`pattern ${id} not found`);
  return await r.json();
}

export async function runDemo(id: string, input: any): Promise<any> {
  const r = await fetch(`/api/execute/${id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input })
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
