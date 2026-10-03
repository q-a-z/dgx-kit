import { useState } from 'react'
import { api } from '../api'
import { usePoll } from '../usePoll'
import { Chart } from '../components/Chart'

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

type Hist = {
  runs: { id: string; started: number; tests: string; context: Record<string, string | number>; metrics: Record<string, number> }[]
  regressions: { metric: string; label: string; unit: string; now: number; usual: number; change: number; points: boolean; since: string[] }[]
}

const CONTEXT_LABELS: Record<string, string> = { image: 'image', image_id: 'image build', dgxkit: 'DGX-kit', driver: 'driver', engine: 'engine', quantization: 'quantization', kv_cache_dtype: 'KV cache type', max_context: 'context', draft_method: 'draft method', speculative_tokens: 'draft tokens' }
/** What differs between two runs' setups, in words (the same rule as the service's, to mark the charts). */
const diff = (a?: Record<string, string | number>, b?: Record<string, string | number>) =>
  Object.entries(CONTEXT_LABELS).filter(([k]) => a?.[k] != null && b?.[k] != null && a[k] !== b[k]).map(([k, l]) => `${l}: ${a![k]} → ${b![k]}`)

/** The finished runs of this model over time: whether the latest got worse than usual, what changed since the run before, and the trend. */
function BenchHistory({ name }: { name: string }) {
  const { data } = usePoll<Hist>(`/api/bench/history?model=${encodeURIComponent(name)}`, 30000)
  if (!data || !data.runs.length) return null
  const runs = data.runs
  const bad = new Set(data.regressions.map((r) => r.metric))
  const times = runs.map((r) => r.started)
  const marks = runs.flatMap((r, i) => (i > 0 && diff(runs[i - 1].context, r.context).length ? [{ t: r.started, label: diff(runs[i - 1].context, r.context)[0].split(':')[0] }] : []))
  const line = (key: string, label: string) => ({ label, values: runs.map((r) => r.metrics[key] ?? null) })
  const since = data.regressions[0]?.since ?? []
  const cols: [string, string, number][] = [['decode_code', 'Decode', 1], ['prefill_tps', 'Prefill', 0], ['conc_total', 'Streams', 0], ['tools_pct', 'Tools %', 0], ['needle_pct', 'Needle %', 0], ['complex_pct', 'Complex %', 0]]
  return (
    <div className="bench-history">
      <h3>History</h3>
      {data.regressions.length > 0 ? (
        <div className="warn">
          <p>▲ The latest run is worse than your usual in:</p>
          <ul>{data.regressions.map((r) => <li key={r.metric}>{r.label}: {f(r.now, 1)} {r.unit} against {f(r.usual, 1)} usually ({r.points ? `${r.change} points` : `${r.change > 0 ? '+' : ''}${r.change}%`})</li>)}</ul>
          <p className="muted small">{since.length ? `Changed since the run before: ${since.join('; ')}.` : 'Nothing recorded changed since the run before (image, DGX-kit, driver, settings): it may be noise, other load on the box, or the model itself.'}</p>
        </div>
      ) : runs.length > 1 && <p className="muted">The latest run is in line with the earlier ones.</p>}
      {runs.length > 1 && (
        <div className="row bench-charts">
          <div className="grow"><span className="lbl">Decode speed, t/s</span>
            <Chart times={times} fit marks={marks} height={110} fmt={(v) => v.toFixed(0)} series={[line('decode_code', 'code'), line('decode_prose', 'prose'), line('decode_complex', 'complex code')]} /></div>
          <div className="grow"><span className="lbl">Prefill, t/s</span>
            <Chart times={times} fit marks={marks} height={110} fmt={(v) => v.toFixed(0)} series={[line('prefill_tps', 'prefill')]} /></div>
        </div>
      )}
      <table className="bench-result"><thead><tr><th>When</th><th>Changed</th>{cols.map(([, l]) => <th key={l}>{l}</th>)}</tr></thead><tbody>
        {runs.slice().reverse().slice(0, 10).map((r, ri, arr) => {
          const prev = arr[ri + 1]
          const changes = prev ? diff(prev.context, r.context) : []
          const latest = ri === 0
          return (
            <tr key={r.id}>
              <td>{new Date(r.started * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</td>
              <td className="small muted" title={changes.join('\n')}>{prev ? (changes.length ? changes.map((c) => c.split(':')[0]).join(', ') : '–') : 'first run'}</td>
              {cols.map(([k, , d]) => <td key={k} className={latest && bad.has(k) ? 'bad' : ''}>{f(r.metrics[k], d)}{latest && bad.has(k) ? ' ▼' : ''}</td>)}
            </tr>
          )
        })}
      </tbody></table>
    </div>
  )
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
      <BenchHistory name={name} />
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
