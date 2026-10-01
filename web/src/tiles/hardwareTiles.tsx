import { fmtBytes, fmtNum, type Snapshot } from '../api'
import { Spark } from '../components/Chart'
import { Status, Tape } from '../components/Tape'
import type { TileConf, TileDef } from './layout'

export type HwCtx = { latest: Snapshot; history: Snapshot[] }

const WINDOW = { key: 'window', label: 'History', default: 5, choices: [[1, '1 min'], [5, '5 min'], [15, '15 min']] as [number, string][] }

export const recent = (h: Snapshot[], minutes: number) => {
  const since = (h.at(-1)?.t ?? 0) - minutes * 60
  let i = h.length
  while (i > 0 && h[i - 1].t >= since) i--
  return h.slice(i)
}
const times = (h: Snapshot[]) => h.map((s) => s.t)
const lvl = (v: number | null | undefined, warn: number, crit: number) =>
  v == null ? '' : crit && v >= crit ? 'crit' : warn && v >= warn ? 'warn' : ''

/** One instrument: the reading, a tape gauge against its limits, and its recent history. */
function readingTile(id: string, title: string, unit: string, pick: (s: Snapshot) => number | null | undefined,
  d: { max: number; warn: number; crit: number; digits?: number }): TileDef<HwCtx> {
  return {
    id, title, w: 3,
    options: [
      { key: 'min', label: 'Scale from', default: 0 },
      { key: 'max', label: 'Scale to', default: d.max },
      { key: 'warn', label: 'Warn at (0 = off)', default: d.warn },
      { key: 'crit', label: 'Critical at (0 = off)', default: d.crit },
      WINDOW,
    ],
    render: ({ latest, history }, o) => {
      const v = pick(latest)
      const h = recent(history, o.window)
      return (
        <>
          <div className="reading-row">
            <div className="reading">{v == null ? '–' : v.toFixed(d.digits ?? 0)}<small>{unit}</small></div>
            <Status level={lvl(v, o.warn, o.crit)} warn={o.warn} crit={o.crit} unit={unit} />
          </div>
          <Tape value={v} min={o.min} max={o.max} warn={o.warn} crit={o.crit} />
          <div className="chart-wrap"><Spark times={times(h)} values={h.map((s) => pick(s) ?? null)} max={o.max} height={34} label={title} fmt={(x) => `${x.toFixed(d.digits ?? 0)} ${unit}`} /></div>
        </>
      )
    },
  }
}

const memSplit = (s: Snapshot) => {
  const m = s.system?.memory
  if (!m) return null
  const cache = m.cached_bytes
  const used = m.total_bytes - m.available_bytes
  // GPU allocations live in the same memory, so they are part of `used`
  const gpu = Math.min(used, (s.gpu?.processes ?? []).reduce((a, p) => a + p.mem_mib * 2 ** 20, 0))
  return { total: m.total_bytes, used, gpu, cpu: Math.max(0, used - gpu), cache, free: Math.max(0, m.total_bytes - used - cache) }
}

const gpuName = (s: Snapshot) => s.gpu?.name?.replace(/^NVIDIA /, '') ?? undefined

export const HW_TILES: TileDef<HwCtx>[] = [
  readingTile('gpu-clock', 'GPU clock', 'MHz', (s) => s.gpu?.sm_clock_mhz, { max: 3003, warn: 0, crit: 0 }),
  {
    id: 'gpu-procs', title: 'On the GPU', sub: ({ latest }) => gpuName(latest), w: 12,
    render: ({ latest }) => {
      const g = latest.gpu
      if (!g) return <p className="muted">NVML isn't available, so GPU processes can't be read.</p>
      const procs = [...g.processes].sort((a, b) => b.mem_mib - a.mem_mib)
      const total = latest.system?.memory.total_bytes ?? 0
      return (
        <>
          {g.events.length > 0 && <div className="events" style={{ marginBottom: 8 }}>{g.events.map((e) => <span key={e} className="pill warn">{e.replaceAll('_', ' ')}</span>)}</div>}
          {procs.length === 0 ? <p className="muted">Nothing is running on the GPU.</p> : (
            <table className="procs">
              <thead><tr><th>Model</th><th>Container</th><th className="num">Context</th><th className="num">PID</th><th className="num">Memory</th></tr></thead>
              <tbody>
                {procs.map((p) => {
                  const live = p.key ? latest.models[p.key] : undefined
                  const ctx = p.ctx ?? (live?.context_tokens as number | undefined)
                  const bytes = p.mem_mib * 2 ** 20
                  return (
                    <tr key={p.pid}>
                      <td><span className="who">{p.model ?? 'Unknown process'}</span>{p.managed && <span className="tag">DGX-kit</span>}</td>
                      <td className="muted">{p.container ?? '–'}</td>
                      <td className="num">{ctx ? fmtNum(ctx) : '–'}</td>
                      <td className="num muted">{p.pid}</td>
                      <td className="num">
                        <span className="memcell">{fmtBytes(bytes)}{total > 0 && <span className="membar"><i style={{ width: `${(100 * bytes) / total}%` }} /></span>}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </>
      )
    },
  },
  {
    id: 'memory', title: 'Unified memory', w: 6, options: [WINDOW],
    render: ({ latest, history }, o) => {
      const m = memSplit(latest)
      if (!m) return <p className="muted">–</p>
      const h = recent(history, o.window)
      const parts = [
        { label: 'GPU', value: m.gpu, color: 'var(--s1)' },
        { label: 'CPU', value: m.cpu, color: 'var(--s2)' },
        { label: 'Cache', value: m.cache, color: 'var(--s3)' },
      ]
      return (
        <>
          <div className="reading">{fmtBytes(m.used)}<span className="of">of {fmtBytes(m.total)} in use</span></div>
          <div className="stack" role="img" aria-label={parts.map((p) => `${p.label} ${fmtBytes(p.value)}`).join(', ')}>
            {parts.map((p) => <div key={p.label} style={{ width: `${(100 * p.value) / m.total}%`, background: p.color }} />)}
          </div>
          <div className="key">
            {parts.map((p) => <span key={p.label}><i style={{ background: p.color }} />{p.label}<b>{fmtBytes(p.value)}</b></span>)}
            <span><i style={{ background: 'var(--track)' }} />Free<b>{fmtBytes(m.free)}</b></span>
          </div>
          <div className="chart-wrap"><Spark times={times(h)} values={h.map((s) => memSplit(s)?.used ?? null)} max={m.total} height={34} label="In use" fmt={(x) => fmtBytes(x)} /></div>
        </>
      )
    },
  },
  {
    id: 'sensors', title: 'Sensors', w: 6,
    render: ({ latest }) => {
      const s = latest.system
      if (!s?.sensors.length) return <p className="muted">No hardware sensors found.</p>
      return (
        <table><tbody>
          {s.sensors.map((x) => <tr key={x.chip + x.label}><td>{x.chip}</td><td className="muted">{x.label}</td><td className="num">{fmtNum(x.value, 1)} {x.unit}</td></tr>)}
        </tbody></table>
      )
    },
  },
]

export const HW_DEFAULTS: TileConf[] = HW_TILES.map((t) => ({ id: t.id, w: t.w, hidden: t.id === 'sensors' || undefined }))
