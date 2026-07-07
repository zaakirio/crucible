export interface Meta {
  db_path: string
  n_runs: number
  n_results: number
  n_judged: number
  models: string[]
}

export interface RunSummary {
  id: number
  model_file: string
  model_name: string | null
  quant: string | null
  lineage: string | null
  hardware: string | null
  llama_cpp_commit: string | null
  ctx: number | null
  ngl: number | null
  repeat: number | null
  model_size_bytes: number | null
  started_at: string | null
  finished_at: string | null
  ppl: number | null
  model_sha256: string | null
  tests_sha256: string | null
  engine_tag: string | null
  server_url: string | null
  crucible_version: string | null
  only_filter: string | null
  status: 'done' | 'open'
  n_results: number
  n_graded: number
  n_passed: number
  n_complied: number
  n_hedged: number
  n_refused: number
  has_hashes: boolean
}

export interface CategorySummary {
  category: string
  n_tests: number
  n_results: number
  n_passed: number
  n_graded: number
  n_refused: number
  n_hedged: number
  n_complied: number
  avg_tps: number | null
}

export interface RunDetail extends RunSummary {
  categories: CategorySummary[]
  flapping: { test_id: string; category: string; reps: number; n_passed: number }[]
  flagged: { category: string; test_id: string; rep: number; flags: string; completion_tokens: number; detail: string }[]
}

export interface ResultRow {
  id: number
  run_id: number
  test_id: string
  category: string
  rep: number
  response: string | null
  passed: number | null
  label: string | null
  detail: string | null
  latency_ms: number | null
  tok_per_sec: number | null
  prompt_tokens: number | null
  completion_tokens: number | null
  flags: string | null
  prompt_text: string | null
  human_label: string | null
  judge_label: string | null
  judge_reason: string | null
  judge_model: string | null
}

export interface ComparePayload {
  a: RunSummary
  b: RunSummary
  rows: {
    category: string
    is_label: boolean
    value_a: string
    value_b: string
    delta: string
    flagged: boolean
  }[]
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status}: ${body.slice(0, 200)}`)
  }
  return res.json()
}

export const api = {
  meta: () => get<Meta>('/api/meta'),
  runs: () => get<RunSummary[]>('/api/runs'),
  run: (id: number) => get<RunDetail>(`/api/runs/${id}`),
  results: (id: number, params: Record<string, string>) => {
    const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v))
    return get<ResultRow[]>(`/api/runs/${id}/results?${qs}`)
  },
  compare: (a: number, b: number) => get<ComparePayload>(`/api/compare?a=${a}&b=${b}`),
}

export function passRate(passed: number, graded: number): number | null {
  return graded > 0 ? passed / graded : null
}

export function fmtDate(iso: string | null): string {
  if (!iso) return '-'
  const d = new Date(iso)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

export function shortHash(h: string | null): string {
  return h ? h.slice(0, 10) : '-'
}

export function fmtBytes(n: number | null): string {
  if (!n) return '-'
  const gb = n / 1e9
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(n / 1e6).toFixed(0)} MB`
}
