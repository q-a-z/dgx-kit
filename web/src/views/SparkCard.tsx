import { fmtNum, type Gateway, type Live, type Snapshot } from '../api'
import { Donut } from '../components/Donut'
import type { Entry } from '../fleet'
import { usePoll } from '../usePoll'

// GB10 GPU power scale; the zones above 80 W are where the box runs hot and loud.
const POWER_MAX = 120

/** The GB10 has no CPU-only sensor; its SoC's thermal zones are the acpitz ones, so the hottest of them stands for the CPU. */
function cpuTemp(latest: Snapshot): number | null {
  const t = (latest.system?.sensors ?? []).filter((x) => x.chip === 'acpitz' && x.unit === 'C').map((x) => x.value)
  return t.length ? Math.max(...t) : null
}

const sum = (xs: Live[], k: string) => xs.reduce((a, x) => a + (typeof x[k] === 'number' ? (x[k] as number) : 0), 0)
const compact = (n: number) => n >= 1e9 ? `${(n / 1e9).toFixed(2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} K` : `${Math.round(n)}`
/** Tokens over the time spent on them, across models: each model's per-request speed weighted by its token count. */
function avgSpeed(xs: Live[], toks: string, speed: string): number | null {
  const parts = xs.filter((x) => typeof x[toks] === 'number' && typeof x[speed] === 'number' && (x[speed] as number) > 0)
  const t = parts.reduce((a, x) => a + (x[toks] as number), 0)
  const secs = parts.reduce((a, x) => a + (x[toks] as number) / (x[speed] as number), 0)
  return secs > 0 ? t / secs : null
}

/** Traffic through the running models: totals since each started, speeds now and on average. */
function TrafficStats({ latest }: { latest: Snapshot }) {
  const up = Object.values(latest.models).filter((x) => x.up)
  if (!up.length) return <div className="cell"><span className="muted small">No model running</span></div>
  const dec = avgSpeed(up, 'gen_tokens_total', 'decode_tps_req')
  const pre = avgSpeed(up, 'prompt_tokens_total', 'prefill_tps_req')
  const waiting = sum(up, 'waiting')
  const stat = (k: string, v: string, sub?: string, tip?: string) =>
    <div title={tip}><dt>{k}</dt><dd><b className="n">{v}</b>{sub && <small className="muted"> {sub}</small>}</dd></div>
  return (
    <dl className="cell traffic">
      {stat('In', compact(sum(up, 'prompt_tokens_total')), 'tokens', 'Prompt tokens since the models started')}
      {stat('Out', compact(sum(up, 'gen_tokens_total')), 'tokens', 'Generated tokens since the models started')}
      {stat('Requests', compact(sum(up, 'requests_total')), undefined, 'Finished requests since the models started')}
      {stat('Decode', fmtNum(sum(up, 'decode_tps')), dec != null ? `/ avg ${fmtNum(dec)} t/s` : 't/s', 'Output tokens per second now, and the average per request')}
      {stat('Prefill', fmtNum(sum(up, 'prefill_tps')), pre != null ? `/ avg ${fmtNum(pre)} t/s` : 't/s', 'Prompt tokens per second now, and the average per request')}
      {stat('Busy', String(sum(up, 'running')), waiting ? `+${waiting} waiting` : undefined, 'Requests running now')}
    </dl>
  )
}

export const TINTS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)']

/** The box itself, in one row: GPU heat, power, free memory and the gateway, each with what it means. */
export function SparkCard({ latest, hwOpen, toggleHw }: {
  latest: Snapshot; history: Snapshot[]; entries: Entry[]; showMemory: boolean; reserved: number; hwOpen: boolean; toggleHw: () => void
}) {
  const g = latest.gpu
  const throttled = !!g?.events.length
  return (
    <section className="system" aria-label="System">
      <div className="sys-head"><h2>System</h2>{g?.name && <span className="muted">{g.name}</span>}<span className="grow" />
        <button className="ghost" aria-expanded={hwOpen} onClick={toggleHw}>{hwOpen ? 'Hide details' : 'CPU, disk, network'}</button>
      </div>
      <div className="sys-row">
        {g && (
          <div className="gauges">
            <Donut label="Temperature" value={g.temp_c} tag="GPU" second={{ value: cpuTemp(latest), tag: 'CPU' }} min={20} max={100} unit="°C"
              zones={[{ upTo: 70, color: 'var(--ok)' }, { upTo: 85, color: 'var(--warn)' }, { upTo: 100, color: 'var(--bad)' }]} />
            <Donut label="GPU power" value={g.power_w} max={POWER_MAX} unit="W"
              zones={[{ upTo: 80, color: 'var(--ok)' }, { upTo: 100, color: 'var(--warn)' }, { upTo: POWER_MAX, color: 'var(--bad)' }]} />
            <Donut label="Load" value={g.util_pct} tag="GPU" second={{ value: latest.system?.cpu_pct?.cpu, tag: 'CPU' }} max={100} unit="%"
              zones={[{ upTo: 75, color: 'var(--s1)' }, { upTo: 90, color: 'var(--warn)' }, { upTo: 100, color: 'var(--bad)' }]} />
            {throttled && <span className="warn small throttle">▲ throttling</span>}
          </div>
        )}
        <TrafficStats latest={latest} />
        <GatewayLine />
      </div>
    </section>
  )
}

/** How clients reach the models, with a link to LiteLLM itself (the clipboard API doesn't exist on plain http). */
export function GatewayLine() {
  const { data } = usePoll<Gateway>('/api/gateway', 10000)
  if (!data) return null
  const url = data.url ?? `${location.protocol}//${location.hostname}:${data.port}/v1`
  const up = data.reachable ?? data.state === 'running'
  return (
    <div className="cell gw">
      <span className="lbl">Gateway</span>
      <span className="gw-row">
        <span className={`gw-status ${up && data.auth !== 'rejected' ? 'ok' : 'warn'}`}>
          <span className="gw-dot" aria-hidden />{data.auth === 'rejected' ? 'Key rejected' : up ? 'Running' : 'Not answering'}
        </span>
        <code title={url}>{url.replace(/^https?:\/\//, '')}</code>
        <a className="gw-link" href={url.replace(/\/v1$/, '/ui/')} target="_blank" rel="noreferrer">Open LiteLLM ↗</a>
      </span>
      {(data.served?.length ? data.served : data.models).length
        ? <span className="served"><span className="lbl">Serves</span>{(data.served?.length ? data.served : data.models).map((m) => <code key={m}>{m}</code>)}</span>
        : <span className="muted small">No models published yet</span>}
    </div>
  )
}

