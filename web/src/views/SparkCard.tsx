import type { ReactNode } from 'react'
import { fmtNum, sum as rateSum, type Live, type Snapshot } from '../api'
import { Donut } from '../components/Donut'

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

/** Speed now against the average: green up when at least as fast, amber level a little slower, red down when well below. Nothing while idle. */
function pace(now: number, avg: number | null) {
  if (!avg || now <= 0) return null
  const r = now / avg
  const [mark, cls, word] = r >= 0.95 ? ['▲', 'ok', 'fast'] : r >= 0.7 ? ['▶', 'warn', 'a little slow'] : ['▼', 'bad', 'slow']
  return <span className={`pace ${cls}`} role="img" aria-label={`${word} against the average`}> {mark}</span>
}

/** Traffic through the running models: totals since each started, speeds now and on average. */
function TrafficStats({ latest }: { latest: Snapshot }) {
  const up = Object.values(latest.models).filter((x) => x.up)
  if (!up.length) return <div className="cell stats"><span className="muted small">No model running</span></div>
  const dec = avgSpeed(up, 'gen_tokens_total', 'decode_tps_req')
  const pre = avgSpeed(up, 'prompt_tokens_total', 'prefill_tps_req')
  const waiting = sum(up, 'waiting')
  const stat = (k: string, v: string, sub?: string, tip?: string, arrow?: ReactNode) =>
    <div title={tip}><dt>{k}</dt><dd><b className="n">{v}</b>{arrow}{sub && <small className="muted"> {sub}</small>}</dd></div>
  return (
    <div className="cell stats">
      <dl className="traffic">
        {stat('In', compact(sum(up, 'prompt_tokens_total')), 'tokens', 'Prompt tokens since the models started')}
        {stat('Out', compact(sum(up, 'gen_tokens_total')), 'tokens', 'Generated tokens since the models started')}
        {stat('Requests', compact(sum(up, 'requests_total')), undefined, 'Finished requests since the models started')}
        {stat('Decode', fmtNum(sum(up, 'decode_tps')), dec != null ? `/ avg ${fmtNum(dec)} t/s` : 't/s', 'Output tokens per second now, and the average per request. The arrow compares now with the average.', pace(sum(up, 'decode_tps'), dec))}
        {stat('Prefill', fmtNum(sum(up, 'prefill_tps')), pre != null ? `/ avg ${fmtNum(pre)} t/s` : 't/s', 'Prompt tokens per second now, and the average per request. The arrow compares now with the average.', pace(sum(up, 'prefill_tps'), pre))}
        {stat('Busy', String(sum(up, 'running')), waiting ? `+${waiting} waiting` : undefined, 'Requests running now')}
      </dl>
    </div>
  )
}

// Where the rings top out; the scale is square-root, so light traffic still shows.
const NET_MAX_MB = 1000
const DISK_MAX_MB = 3000
const mb = (bps: number | null | undefined) => (bps == null ? null : bps / 1e6)
/** The main interface's rate, else the sum over all of them (as the Network tile does). */
function net(latest: Snapshot, key: 'net_rx_bps' | 'net_tx_bps'): number | undefined {
  const r = latest.system?.[key], iface = latest.system?.net_iface
  return r ? (iface && iface in r ? r[iface] : rateSum(r)) : undefined
}

/** The CPU's cores as small bars, the same view as the CPU tile at the bottom, sized to stand beside the ring gauges. */
function CoreGauge({ latest }: { latest: Snapshot }) {
  const s = latest.system
  const cores = Object.entries(s?.cpu_pct ?? {}).filter(([k]) => k !== 'cpu').sort(([a], [b]) => Number(a.slice(3)) - Number(b.slice(3)))
  if (!cores.length) return null
  return (
    <div className="donut core-gauge">
      <div className="core-grid" role="group" aria-label="Load per CPU core">
        {cores.map(([k, v]) => (
          <div key={k} className={`core ${v >= 97 ? 'crit' : v >= 85 ? 'warn' : ''}`} title={`Core ${k.slice(3)}: ${fmtNum(v)}% at ${fmtNum(s?.cpu_freq_mhz[k])} MHz`}>
            <i style={{ height: `${Math.max(v, 4)}%` }} />
          </div>
        ))}
      </div>
      <span className="lbl">CPU cores</span>
    </div>
  )
}

export const TINTS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)']

/** The box itself, in one row: GPU heat, power, free memory and the gateway, each with what it means. */
export function SparkCard({ latest }: { latest: Snapshot }) {
  const g = latest.gpu
  const throttled = !!g?.events.length
  return (
    <section className="system" aria-label="System">
      <div className="sys-head"><h2>System</h2>{g?.name && <span className="muted">{g.name}</span>}<span className="grow" /><h2 className="stats-title">Stats</h2>
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
            <Donut label="Network" value={mb(net(latest, 'net_rx_bps'))} tag="In" second={{ value: mb(net(latest, 'net_tx_bps')), tag: 'Out' }} max={NET_MAX_MB} unit="MB/s" curve fine />
            <Donut label="Disk" value={mb(rateSum(latest.system?.disk_read_bps))} tag="Read" second={{ value: mb(rateSum(latest.system?.disk_write_bps)), tag: 'Write' }} max={DISK_MAX_MB} unit="MB/s" curve fine />
            <CoreGauge latest={latest} />
            {throttled && <span className="warn small throttle">▲ throttling</span>}
          </div>
        )}
        <TrafficStats latest={latest} />
      </div>
    </section>
  )
}
