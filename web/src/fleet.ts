import { fmtDateTime, fmtUptime, type Download, type Live, type ModelRow, type Snapshot } from './api'
import { usePoll } from './usePoll'

export type External = { name: string; engine: string; port: number; model: string | null; served_name: string | null; draft: string | null; image: string; live?: Live | null }

/** Downloading or paused mid-download. */
export const fetching = (s: string) => s === 'downloading' || s === 'paused'

/** One model on this box, whoever started it. */
export type Entry = {
  name: string
  managed: boolean
  row?: ModelRow
  ext?: External
  running: boolean
  live?: Live
  download?: Download
  memBytes: number | null
  /** running | starting | preparing (fetching its image) | down | stopped | exited (crashed) | restarting | downloading | failed | missing */
  state: string
}

export const ENGINES: Record<string, string> = { vllm: 'vLLM', sglang: 'SGLang', llamacpp: 'llama.cpp' }
export const base = (p: string | null | undefined) => (p ? p.replace(/\/+$/, '').split('/').pop() ?? p : null)
const ACTIVE = new Set(['queued', 'running', 'paused', 'failed'])

export function stateOf(e: Omit<Entry, 'state'>): string {
  const d = e.download
  if (e.row?.preparing && !e.running) return 'preparing'
  if (d && d.state === 'failed') return 'failed'
  if (d && d.state === 'paused') return 'paused'
  if (d && ACTIVE.has(d.state)) return 'downloading'
  if (e.running) return e.live ? (e.live.up ? 'running' : e.live.error && e.live.was_up ? 'down' : 'starting') : 'starting'  // refusing connections before its first answer is just loading
  const c = e.row?.container?.state
  if (c === 'created') return 'starting'
  if (c === 'restarting') return 'restarting'
  // A container that was stopped on purpose is left behind, and vLLM exits with 0 or 1 when stopped, so the exit code
  // can't tell a stop from a crash. The service says it stopped cleanly when it made the stop itself or the engine
  // logged its own "Application shutdown complete"; a clean exit code (0, 143) counts too. Anything else, or a
  // container Docker keeps restarting, is still a problem.
  if (c === 'exited' && (e.row?.container?.stopped_cleanly || [0, 143].includes(e.row?.container?.exit_code ?? -1))) return 'stopped'
  if (c && c !== 'running') return 'exited'
  if (e.row && !e.row.downloaded) return e.row.path ? 'nofiles' : 'missing'  // a folder on disk that isn't there vs a repo not downloaded yet
  return 'stopped'
}

export const STATE_LABEL: Record<string, string> = {
  running: 'Serving', starting: 'Starting', down: 'Not answering', stopped: 'Stopped', exited: 'Crashed', restarting: 'Crashing, restarting',
  preparing: 'Fetching image', downloading: 'Downloading', paused: 'Download paused', failed: 'Download failed', missing: 'Not downloaded', nofiles: 'Weights not found',
}
/** What the saved settings say, in one line: the starting text for a note, so it can't disagree with the config. */
export const describeModel = (r: ModelRow) =>
  [base(r.repo), [ENGINES[r.engine] ?? r.engine, r.quantization].filter(Boolean).join(' '),
    [r.draft_repo && r.num_speculative_tokens ? `draft n=${r.num_speculative_tokens}` : '', r.kv_cache_dtype && `${r.kv_cache_dtype} KV`, r.max_context ? `ctx ${r.max_context}` : '']
      .filter(Boolean).join(', ')].filter(Boolean).join(' · ')

/** "Serving", and for a running model how long: "Serving for 3 h 12 min". */
export const stateText = (e: Entry) => {
  const label = STATE_LABEL[e.state] ?? e.state
  const started = e.row?.container?.started
  return e.state === 'running' && started ? `${label} for ${fmtUptime(Date.now() / 1000 - started)}` : label
}
/** When it started, for the hover on that text. */
export const stateTitle = (e: Entry) => (e.state === 'running' && e.row?.container?.started ? `Running since ${fmtDateTime(e.row.container.started, 'medium')}` : undefined)

export const STATE_TONE: Record<string, string> = { running: 'ok', starting: 'pending', down: 'bad', exited: 'bad', restarting: 'bad', failed: 'bad', nofiles: 'bad', downloading: 'pending', paused: 'pending', preparing: 'pending' }

/** Every model DGX-kit knows about or can see running, with its live numbers and GPU memory. */
export function useFleet(latest: Snapshot | null) {
  const models = usePoll<ModelRow[]>('/api/models', 3000)
  const running = usePoll<External[]>('/api/running', 5000)
  const downloads = usePoll<Download[]>('/api/downloads', 2000)
  const procs = latest?.gpu?.processes ?? []
  // A DGX-kit model owns only the processes in its own container; matching by served name would hand
  // it the memory of someone else's model that happens to share the name (and make a crashed one look alive).
  const mem = (name: string, managed: boolean) => {
    const mine = procs.filter((p) => p.key === name ? !!p.managed === managed : !managed && !p.key && p.model === name)
    return mine.length ? mine.reduce((a, p) => a + p.mem_mib * 2 ** 20, 0) : null
  }
  const dl = (repo: string | null | undefined, draft?: string | null) =>
    (downloads.data ?? []).find((d) => ACTIVE.has(d.state) && (d.repo === repo || (!!draft && d.repo === draft)))
  const make = (e: Omit<Entry, 'state'>): Entry => ({ ...e, state: stateOf(e) })

  const entries: Entry[] = [
    ...(models.data ?? []).map((row) => make({
      name: row.name, managed: true, row, running: row.container?.state === 'running',
      live: latest?.models[row.name] ?? row.live ?? undefined, download: dl(row.repo, row.draft_repo), memBytes: mem(row.name, true),
    })),
    ...(running.data ?? []).map((ext) => make({ name: ext.name, managed: false, ext, running: true, live: latest?.models[ext.name] ?? ext.live ?? undefined, memBytes: mem(ext.name, false) })),
  ]
  // A model can be live in the stream before the lists load.
  for (const [n, live] of Object.entries(latest?.models ?? {}))
    if (!entries.some((e) => e.name === n)) entries.push(make({ name: n, managed: !live.external, running: true, live, memBytes: mem(n, !live.external) }))
  // Downloads for models not added yet (e.g. a draft on its own).
  const orphans = (downloads.data ?? []).filter((d) => ACTIVE.has(d.state) && !entries.some((e) => e.download === d))

  return {
    entries, orphans, loaded: !!models.data,
    reload: () => { models.reload(); running.reload(); downloads.reload() },
  }
}

const DROP = /^(instruct|it|chat|hf|gguf|fp8|fp16|bf16|nvfp4|mxfp4|int4|int8|awq|gptq|w4a16|w8a8|exl2|q\d.*|\d{4}|v\d+(\.\d+)*-?hf)$/i
const SPECIAL: Record<string, string> = { gpt: 'GPT', oss: 'OSS', glm: 'GLM', llm: 'LLM', vl: 'VL', moe: 'MoE' }

/** A readable name for people: "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8" becomes "Qwen3 Coder 30B A3B". */
export function prettyName(e: Entry): string {
  const src = base(e.row?.repo ?? e.ext?.model ?? e.ext?.served_name) ?? e.name
  const words = src.replace(/\.gguf$/i, '').split(/[-_\s]+/).filter((w) => w && !DROP.test(w))
  if (!words.length) return e.name
  return words.map((w) => SPECIAL[w.toLowerCase()] ?? (/^\d+(\.\d+)?[bmk]$/i.test(w) || /^a\d+b$/i.test(w) ? w.toUpperCase()
    : /^[a-z]/.test(w) ? w[0].toUpperCase() + w.slice(1) : w)).join(' ')
}
