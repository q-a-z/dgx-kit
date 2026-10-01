import { fmtBytes, fmtNum, type Snapshot } from '../api'
import type { TileConf, TileDef } from './layout'

export type HwCtx = { latest: Snapshot; history: Snapshot[] }

export const recent = (h: Snapshot[], minutes: number) => {
  const since = (h.at(-1)?.t ?? 0) - minutes * 60
  let i = h.length
  while (i > 0 && h[i - 1].t >= since) i--
  return h.slice(i)
}
const gpuName = (s: Snapshot) => s.gpu?.name?.replace(/^NVIDIA /, '') ?? undefined

export const HW_TILES: TileDef<HwCtx>[] = [
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
