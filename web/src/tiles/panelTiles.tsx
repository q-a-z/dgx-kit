import { fmtNum, fmtTime, type Live, type Snapshot } from '../api'
import { Chart } from '../components/Chart'
import { KV_WARN } from '../health'
import type { TileConf, TileDef } from './layout'
import { recent } from './hardwareTiles'
import { MODEL_TILES, type ModelCtx } from './modelTiles'

const WINDOW = { key: 'window', label: 'History', default: 15, choices: [[5, '5 min'], [15, '15 min'], [30, '30 min']] as [number, string][] }
const num = (l: Live | undefined, k: string) => (typeof l?.[k] === 'number' ? (l[k] as number) : null)
const col = (h: Snapshot[], name: string, k: string, scale = 1) => h.map((s) => {
  const v = s.models[name]?.[k]
  return typeof v === 'number' ? v * scale : null
})

const mean = (xs: (number | null)[]) => {
  const v = xs.filter((x): x is number => x != null)
  return v.length >= 3 ? v.reduce((a, b) => a + b, 0) / v.length : null
}

/** Average over the 2 minutes before the latest restart or change against everything since, if both sides have data. */
function beforeAfter(c: ModelCtx, times: number[], values: (number | null)[]) {
  const m = [...(c.marks ?? [])].reverse().find((x) => x.t >= (times[0] ?? Infinity) && x.t <= (times.at(-1) ?? 0))
  if (!m) return null
  // Skip the few seconds a restart takes, so the gap doesn't count as a drop.
  const before = mean(values.filter((_, i) => times[i] < m.t && times[i] >= m.t - 120))
  const after = mean(values.filter((_, i) => times[i] > m.t + 5))
  if (before == null || after == null) return null
  return { label: m.label, at: fmtTime(m.t), before, after }
}

/** One number that answers one question, what it's measured against, and its recent history with restarts marked. */
function kpi(id: string, title: string, d: {
  key: string; scale?: number; unit: string; digits?: number; max?: number
  context?: (c: ModelCtx) => string | undefined; abnormal?: (v: number, c: ModelCtx) => boolean
}): TileDef<ModelCtx> {
  return {
    id, title, w: 6, options: [WINDOW],
    render: (c, o) => {
      const raw = num(c.live, d.key)
      const v = raw == null ? null : raw * (d.scale ?? 1)
      const bad = v != null && !!d.abnormal?.(v, c)
      const h = recent(c.history, o.window)
      const times = h.map((s) => s.t)
      const values = col(h, c.name, d.key, d.scale)
      const ba = beforeAfter(c, times, values)
      const f = (x: number) => fmtNum(x, d.digits ?? (x < 10 ? 1 : 0))
      return (
        <>
          <div className={`kpi ${bad ? 'warn' : ''}`}>
            <span className="kpi-v">{bad && <span aria-hidden>▲ </span>}{v == null ? '–' : fmtNum(v, d.digits ?? (v < 10 ? 1 : 0))}</span>
            <span className="kpi-u">{d.unit}{d.context?.(c) ? ` · ${d.context(c)}` : ''}</span>
          </div>
          {ba && <div className="ba">{ba.label} {ba.at}: <b>{f(ba.before)} → {f(ba.after)}</b> {d.unit}{ba.before > 0 && Math.round((100 * (ba.after - ba.before)) / ba.before) !== 0 ? ` (${ba.after > ba.before ? '+' : ''}${Math.round((100 * (ba.after - ba.before)) / ba.before)}%)` : ''}</div>}
          <div className="chart-wrap">
            <Chart marks={c.marks} times={times} height={64} max={d.max} fmt={(x) => fmtNum(x, x < 10 ? 1 : 0)}
              series={[{ label: title, values }]} />
          </div>
        </>
      )
    },
  }
}

export const PANEL_TILES: TileDef<ModelCtx>[] = [
  kpi('k-decode', 'Decode speed', { key: 'decode_tps', unit: 'tok/s', context: (c) => (num(c.live, 'decode_tps_req') != null ? `${fmtNum(num(c.live, 'decode_tps_req'), 0)} per request` : undefined) }),
  kpi('k-prefill', 'Prefill speed', { key: 'prefill_tps', unit: 'tok/s', context: (c) => (num(c.live, 'prefill_tps_avg') != null ? `${fmtNum(num(c.live, 'prefill_tps_avg'), 0)} average` : undefined) }),
  kpi('k-ttft', 'First token, p95', {
    key: 'ttft_p95_s', scale: 1000, unit: 'ms',
    context: (c) => (c.slo?.ttft ? `target ${fmtNum(c.slo.ttft * 1000)}` : undefined),
    abnormal: (v, c) => !!c.slo?.ttft && v > c.slo.ttft * 1000,
  }),
  kpi('k-queue', 'Waiting requests', {
    key: 'waiting', unit: 'waiting', digits: 0,
    context: (c) => `${fmtNum(num(c.live, 'running'))} running`,
    abnormal: (v) => v > 0,
  }),
  kpi('k-kv', 'KV cache', { key: 'kv_used_pct', unit: '%', digits: 0, max: 100, context: () => `warn at ${KV_WARN}`, abnormal: (v) => v >= KV_WARN }),
  ...MODEL_TILES.map((t) => ({ ...t, id: `m-${t.id}`, w: 12 })),
]

/** The four questions first; the detailed tiles are one click away under Add. */
export const PANEL_DEFAULTS: TileConf[] = PANEL_TILES.map((t) => ({ id: t.id, w: t.w, hidden: t.id.startsWith('m-') }))
