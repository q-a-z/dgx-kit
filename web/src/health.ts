import { fmtBytes, fmtNum, type Gateway, type Snapshot } from './api'
import { prettyName, type Entry } from './fleet'

/** Something that needs a person. `model` names the object it is about, so the fix sits next to it. */
export type Alert = { level: 'bad' | 'warn'; text: string; short: string; model?: string; advice?: string }

const TEMP_WARN = 85
const FREE_WARN = 5 * 2 ** 30
export const KV_WARN = 90

/** Most urgent first. Normal states never appear here. */
export function alertsFor(entries: Entry[], latest: Snapshot | null, gateway: Gateway | undefined, readonly: boolean): Alert[] {
  const out: Alert[] = []
  for (const e of entries) {
    const n = prettyName(e)
    if (e.state === 'exited') out.push({ level: 'bad', model: e.name, short: `${n} stopped`, text: `${n} stopped on its own.`, advice: 'The last lines of its log usually say why.' })
    if (e.state === 'down') out.push({ level: 'bad', model: e.name, short: `${n} not answering`, text: `${n} is running but not answering requests.`, advice: 'Check its log; a restart often clears a stuck engine.' })
    if (e.state === 'failed') out.push({ level: 'bad', model: e.name, short: `${n} download failed`, text: `The download for ${n} failed${e.download?.error ? `: ${e.download.error}` : '.'}` })
    const l = e.live
    if (e.state === 'running' && l) {
      const waiting = l.waiting ?? 0
      const kv = l.kv_used_pct ?? 0
      if (waiting > 0 || kv >= KV_WARN) {
        const parts = [waiting > 0 && `${fmtNum(waiting)} request${waiting === 1 ? '' : 's'} waiting`, kv >= KV_WARN && `KV cache ${fmtNum(kv)}% full`].filter(Boolean)
        out.push({ level: 'warn', model: e.name, short: `${n} is queueing`, text: `${n}: ${parts.join(', ')}.`,
          advice: kv >= KV_WARN ? 'A shorter max context or fp8 KV cache gives it room for more requests at once.' : 'More requests arrive than it can run at once.' })
      }
    }
  }
  const g = latest?.gpu
  if (g?.events.length) out.push({ level: 'warn', short: 'GPU throttling', text: `The GPU is slowing itself down: ${g.events.map((x) => x.replaceAll('_', ' ')).join(', ')}.` })
  if (g?.temp_c != null && g.temp_c >= TEMP_WARN) out.push({ level: 'warn', short: `GPU ${fmtNum(g.temp_c)} °C`, text: `The GPU is at ${fmtNum(g.temp_c)} °C, close to where it throttles.` })
  const m = latest?.system?.memory
  if (m && m.available_bytes < FREE_WARN) out.push({ level: 'bad', short: 'Memory almost full', text: `Only ${fmtBytes(m.available_bytes)} of memory is free. Starting another model will fail.` })
  if (gateway?.problem && !readonly) out.push({ level: 'warn', short: 'Gateway problem', text: `The gateway isn’t publishing models: ${gateway.problem}` })
  return out.sort((a, b) => (a.level === b.level ? 0 : a.level === 'bad' ? -1 : 1))
}
