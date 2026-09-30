import { fmtBytes, type Gateway, type Snapshot } from '../api'
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

export const TINTS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)']

/** The box itself, in one row: GPU heat, power, free memory and the gateway, each with what it means. */
export function SparkCard({ latest, reserved, hwOpen, toggleHw }: {
  latest: Snapshot; history: Snapshot[]; entries: Entry[]; showMemory: boolean; reserved: number; hwOpen: boolean; toggleHw: () => void
}) {
  const g = latest.gpu
  const m = latest.system?.memory
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
        {m && (
          <div className="cell">
            <span className="lbl">Memory free</span>
            <span><b className="n">{fmtBytes(m.available_bytes)}</b> <span className="muted">of {fmtBytes(m.total_bytes)}</span></span>
            <span className="muted small">{reserved > 0 ? `${fmtBytes(reserved)} wanted by downloads` : 'nothing waiting for room'}</span>
          </div>
        )}
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

