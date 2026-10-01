export type Gpu = {
  name: string | null
  sm_clock_mhz: number | null
  sm_clock_max_mhz: number | null
  util_pct: number | null
  temp_c: number | null
  power_w: number | null
  events: string[]
  processes: GpuProcess[]
}

export type GpuProcess = { pid: number; mem_mib: number; model?: string | null; ctx?: number | null; container?: string | null; key?: string | null; managed?: boolean }

export type Sensor = { chip: string; label: string; unit: string; value: number }

export type System = {
  memory: { total_bytes: number; used_bytes: number; available_bytes: number; cached_bytes: number; swap_used_bytes: number }
  load: number[]
  cpu_freq_mhz: Record<string, number>
  sensors: Sensor[]
  cpu_pct?: Record<string, number>
  net_rx_bps?: Record<string, number>
  net_tx_bps?: Record<string, number>
  disk_read_bps?: Record<string, number>
  disk_write_bps?: Record<string, number>
  disk_busy_pct?: Record<string, number>
  net_iface?: string | null
  cpu_model?: string | null
  /** The box, from the firmware: a short name, and the parts it is made of. */
  machine?: { name: string; vendor: string; product: string; family: string; bios: string } | null
}

export type Live = {
  up: boolean
  was_up?: boolean  // has answered at least once since it was started
  error?: string
  context_tokens?: number
  kv_pool_tokens?: number
  max_concurrency?: number
  running?: number
  waiting?: number
  kv_used_pct?: number
  decode_tps?: number
  prefill_tps?: number
  draft_acceptance?: number
  draft_accept_len?: number
  prefix_hit_rate?: number
  ttft_p50_s?: number
  ttft_p95_s?: number
  itl_p50_s?: number
  itl_p95_s?: number
  ttft_goodput?: number
  external?: boolean
  [metric: string]: number | boolean | string | null | undefined
}

export type Snapshot = { t: number; gpu: Gpu | null; system: System | null; models: Record<string, Live> }

export type Recipe = {
  name: string
  repo: string
  engine: string
  image: string | null
  quantization: string | null
  gguf_file: string | null
  draft_repo: string | null
  draft_method: string | null
  num_speculative_tokens: number
  kv_cache_dtype: string
  max_context: number | null
  min_context: number
  min_concurrency: number
  extra_args: string[]
  quick?: { decode_mean_tps?: number | null; prefill_mean_tps?: number | null; skipped?: string; error?: string } | null  // the speed check taken when it started
  options?: string | null  // vLLM models: every engine setting as text, one flag per line
  speculative_extra?: Record<string, unknown>
  env?: Record<string, string>
  notes?: string
  path?: string | null
  draft_path?: string | null
  publish: boolean
  weights_bytes: number
  draft_weights_bytes: number
  config?: Record<string, unknown>
}

export type ModelRow = Recipe & {
  container: { state: string; port: number; exit_code?: number | null; stopped_cleanly?: boolean } | null
  live: Live | null
  downloaded: boolean
}

export type Plan = { context_tokens: number; kv_pool_tokens: number; kv_bytes: number; concurrency: number; fits: boolean; reason: string }

export class ApiError extends Error {
  problems: string[]
  constructor(message: string, problems: string[] = []) {
    super(message)
    this.problems = problems
  }
}

export async function api<T>(path: string, init?: RequestInit & { json?: unknown }): Promise<T> {
  const { json, ...rest } = init ?? {}
  const res = await fetch(path, {
    ...rest,
    headers: json !== undefined ? { 'content-type': 'application/json' } : undefined,
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  })
  const body = await res.json().catch(() => null)
  if (res.status === 401 && path !== '/api/login') window.dispatchEvent(new Event('dgxkit:signed-out'))
  if (!res.ok) {
    const detail = body?.detail
    const problems: string[] = detail?.problems ?? []
    throw new ApiError(problems[0] ?? (typeof detail === 'string' ? detail : `${res.status} ${res.statusText}`), problems)
  }
  return body as T
}

export const fmtBytes = (b: number | null | undefined) => {
  if (b == null) return '–'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = b
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`
}

export const fmtNum = (n: number | null | undefined, digits = 0) =>
  n == null ? '–' : n.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })

export const fmtMs = (s: number | null | undefined) => (s == null ? '–' : `${Math.round(s * 1000)} ms`)

export type Download = { repo: string; state: string; error: string | null; bytes: number; total_bytes: number; pct: number | null; bytes_per_s: number | null }
export type ImageJob = { kind: string; image: string; state: string; error: string | null; tail: string[] }
export type ImageBuild = { id: string; image: string; about: string; ready: boolean; patches: string[]; job: ImageJob | null }
export type Image = { engine: string; image: string; default: string; ready: boolean; builds: ImageBuild[]; job: ImageJob | null; local?: string[]; source?: 'default' | 'chosen' | 'found' }
export type Action = { t: number; action: string; detail: string }

export const fmtRate = (b: number | null | undefined) => (b == null ? '–' : `${fmtBytes(b)}/s`)
export const sum = (r: Record<string, number> | undefined) => (r ? Object.values(r).reduce((a, b) => a + b, 0) : undefined)
export type Gateway = { state: string | null; port: number; image: string; problem: string | null; models: string[]; external?: string | null; db?: string | null; extra_env?: string[]; extra_env_file?: string; key_ready?: boolean; url?: string; reachable?: boolean; auth?: string | null; served?: string[]; error?: string }

/** Copy text to the clipboard. The clipboard API only exists on https and localhost, and the dashboard is often plain http
 *  on the LAN, so fall back to a hidden text box and the old copy command. Resolves false when neither worked. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); return true }
  } catch { /* fall through to the old way */ }
  const box = document.createElement('textarea')
  box.value = text
  box.setAttribute('readonly', '')
  box.style.cssText = 'position:fixed;top:0;left:0;opacity:0'
  document.body.appendChild(box)
  box.select()
  try { return document.execCommand('copy') } catch { return false } finally { box.remove() }
}
