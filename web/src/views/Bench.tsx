import { useState } from 'react'
import { api } from '../api'
import { usePoll } from '../usePoll'

type Res = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any -- bench.py's own json, shown as it comes
type Run = { id: string; model: string; state: 'running' | 'done' | 'failed'; tests: string; started: number; tail?: string[]; result?: Res | null }

const QUICK = ['decode', 'conc', 'tools', 'sanity']
const mean = (a?: number[]) => (a?.length ? a.reduce((x, y) => x + y, 0) / a.length : null)
const f = (v: number | null | undefined, d = 0, unit = '') => (v == null ? '–' : `${v.toFixed(d)}${unit}`)

/** The numbers worth reading from bench.py's json; a test that failed shows its error instead. */
function summary(r: Res): [string, string][] {
  const rows: [string, string][] = []
  const err = (k: string) => (r[k]?.error ? `failed: ${r[k].error}` : null)
  if (r.decode) rows.push(['Decode, code', err('decode') ?? `${f(mean(r.decode.code_temp0?.tps), 1)} t/s · draft accepts ${f(mean(r.decode.code_temp0?.acc), 2)} tokens/step`])
  if (r.complex) rows.push(['Decode, complex', err('complex') ?? `${f(mean(r.complex.temp0?.tps), 1)} t/s`])
  if (r.hardcore) rows.push(['Decode, hardcore', err('hardcore') ?? `${f(mean(r.hardcore.tps), 1)} t/s`])
  if (r.conc) rows.push([`${r.conc.streams ?? ''} concurrent streams`, err('conc') ?? `${f(r.conc.aggregate, 1)} t/s total · ${(r.conc.per_stream ?? []).map((x: number) => f(x, 0)).join(' + ')} each`])
  if (r.prefill) rows.push(['Prefill', err('prefill') ?? (Array.isArray(r.prefill) ? `${f(mean(r.prefill.map((x: Res) => x.prefill_tps)), 0)} t/s · first token after ${f(mean(r.prefill.map((x: Res) => x.ttft)), 1, ' s')}` : '–')])
  if (r.stall) rows.push(['Decode stall while prefilling', err('stall') ?? (Array.isArray(r.stall) ? `${f(Math.max(...r.stall.map((x: Res) => x.gap_during_max_ms)), 0, ' ms')} worst gap (${f(mean(r.stall.map((x: Res) => x.gap_before_ms)), 0, ' ms')} normally)` : '–')])
  if (r.needle) rows.push(['Needle in a long prompt', err('needle') ?? (Array.isArray(r.needle) ? `${r.needle.filter((x: Res) => x.ok).length} of ${r.needle.length} found · ${f(mean(r.needle.map((x: Res) => x.ptoks)) ?? null, 0)} tokens` : '–')])
  if (r.tools) rows.push(['Tool calls', err('tools') ?? `${r.tools.valid} of ${r.tools.total} valid`])
  if (r.sanity) rows.push(['Sanity', err('sanity') ?? (r.sanity.empty ? 'empty answer' : `${r.sanity.words} words, ${f(r.sanity.repeated_8gram_ratio, 3)} repeated`)])
  return rows
}

/** Runs tools/bench.py against the running model and shows what came out. */
export function Bench({ name, ready, readonly }: { name: string; ready: boolean; readonly: boolean }) {
  const runs = usePoll<Run[]>(`/api/bench?model=${encodeURIComponent(name)}`, 3000)
  const [open, setOpen] = useState<string | null>(null)
  const [detail, setDetail] = useState<Run | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const list = runs.data ?? []
  const shown = open ?? list[0]?.id ?? null
  const live = usePoll<Run>(shown ? `/api/bench/${encodeURIComponent(shown)}` : '/api/bench/none', 3000)
  const d = live.data && live.data.id === shown ? live.data : detail && detail.id === shown ? detail : null
  const start = (tests?: string[]) => api<Run>(`/api/models/${encodeURIComponent(name)}/bench`, { method: 'POST', json: tests ? { tests } : {} })
    .then((r) => { setMsg(null); setOpen(r.id); setDetail(r); runs.reload() })
    .catch((e: Error) => setMsg(e.message))
  const stop = (id: string) => api(`/api/bench/${encodeURIComponent(id)}/stop`, { method: 'POST' }).then(() => runs.reload()).catch((e: Error) => setMsg(e.message))
  const busy = list.some((r) => r.state === 'running')
  const when = (t: number) => new Date(t * 1000).toLocaleString()
  return (
    <div className="bench">
      <div className="row">
        <button className="primary" disabled={readonly || !ready || busy} onClick={() => start()} title="Every test; needs 30 minutes or more with long contexts">Full battery</button>
        <button disabled={readonly || !ready || busy} onClick={() => start(QUICK)} title={QUICK.join(', ')}>Quick run</button>
        <span className="muted small">{ready ? 'Runs on this model only; stop other models’ traffic first, the script refuses if they are busy.' : 'Start the model and wait until it is serving.'}</span>
      </div>
      {msg && <p className="bad">{msg}</p>}
      {list.length === 0 && <p className="muted">No runs yet.</p>}
      {list.length > 0 && (
        <ul className="bench-runs">
          {list.map((r) => (
            <li key={r.id}>
              <button className={`link ${r.id === shown ? 'on' : ''}`} onClick={() => setOpen(r.id)}>{when(r.started)}</button>
              <span className={r.state === 'done' ? '' : r.state === 'failed' ? 'bad' : 'muted'}> {r.state === 'running' ? 'running…' : r.state}</span>
              <span className="muted small"> · {r.tests.split(',').length === 9 ? 'full battery' : r.tests}</span>
            </li>
          ))}
        </ul>
      )}
      {d?.state === 'running' && (
        <>
          <pre className="log">{(d.tail ?? []).join('\n') || 'starting…'}</pre>
          <button onClick={() => stop(d.id)}>Stop run</button>
        </>
      )}
      {d?.state === 'failed' && <pre className="log bad">{(d.tail ?? []).slice(-12).join('\n')}</pre>}
      {d?.state === 'done' && d.result && (
        <table className="bench-result"><tbody>
          {summary(d.result).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
        </tbody></table>
      )}
    </div>
  )
}
