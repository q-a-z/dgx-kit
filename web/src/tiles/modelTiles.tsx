import { fmtNum, type Live, type Snapshot } from '../api'
import { Chart, type Mark } from '../components/Chart'
import { Hero, Metric, SloEditor, type Slo } from '../components/ModelParts'
import type { TileConf, TileDef } from './layout'
import { recent } from './hardwareTiles'

export type { Slo }
export type ModelCtx = { marks?: Mark[]; name: string; live: Live | undefined; history: Snapshot[]; stat: string; slo: Slo | null; setSlo: (s: Slo) => void; readonly: boolean }

const WINDOW = { key: 'window', label: 'History', default: 5, choices: [[1, '1 min'], [5, '5 min'], [15, '15 min']] as [number, string][] }
const STAT_LABEL: Record<string, string> = { avg: 'average', p50: 'median', p95: 'p95', p99: 'p99' }

const num = (l: Live | undefined, k: string) => {
  const v = l?.[k]
  return typeof v === 'number' ? v : null
}
const col = (h: Snapshot[], name: string, k: string, scale = 1) =>
  h.map((s) => {
    const v = s.models[name]?.[k]
    return typeof v === 'number' ? v * scale : null
  })
/** Seconds as [number, unit], in ms below 10 s. */
const dur = (s: number | null): [string, string | undefined] =>
  s == null ? ['–', undefined] : s >= 10 ? [fmtNum(s, 1), 's'] : [fmtNum(s * 1000, s < 0.01 ? 1 : 0), 'ms']
const pct = (f: number | null, d = 1) => (f == null ? '–' : fmtNum(f * 100, d))
const tps = (v: number | null) => fmtNum(v, v != null && v < 100 ? 1 : 0)

function throughput(kind: 'prefill' | 'decode', title: string, accent: boolean): TileDef<ModelCtx> {
  const total = kind === 'prefill' ? 'prompt_tokens_total' : 'gen_tokens_total'
  return {
    id: kind, title, w: 2, options: [WINDOW],
    render: ({ marks, name, live, history }, o) => {
      const h = recent(history, o.window)
      return (
        <>
          <Hero label={`Average, last ${live?.window_s ?? 60} s`} value={tps(num(live, `${kind}_tps_avg`))} unit="tok/s" accent={accent} />
          <div className="metrics">
            <Metric label="Now" value={tps(num(live, `${kind}_tps`))} unit="tok/s" />
            <Metric label="Per request" value={tps(num(live, `${kind}_tps_req`))} unit="tok/s" />
            <Metric label={kind === 'prefill' ? 'Tokens read' : 'Tokens written'} value={fmtNum(num(live, total))} />
          </div>
          <div className="chart-wrap">
            <Chart marks={marks} times={h.map((s) => s.t)} height={90} fmt={(v) => fmtNum(v)}
              series={[{ label: 'Now', values: col(h, name, `${kind}_tps`) }, { label: 'Average', values: col(h, name, `${kind}_tps_avg`) }]} />
          </div>
        </>
      )
    },
  }
}

export const MODEL_TILES: TileDef<ModelCtx>[] = [
  throughput('decode', 'Decode', false),
  throughput('prefill', 'Prefill', false),
  {
    id: 'latency', title: 'Latency', sub: ({ stat }) => STAT_LABEL[stat] ?? stat, w: 2, options: [WINDOW],
    render: ({ marks, name, live, history, stat }, o) => {
      const k = (m: string) => `${m}_${stat}_s`
      const h = recent(history, o.window)
      const [tv, tu] = dur(num(live, k('ttft')))
      const cell = (label: string, m: string) => {
        const [v, u] = dur(num(live, k(m)))
        return <Metric label={label} value={v} unit={u} />
      }
      return (
        <>
          <Hero label="Time to first token" value={tv} unit={tu} />
          <div className="metrics">
            {cell('Between tokens', 'itl')}
            {cell('Per output token', 'tpot')}
            {cell('Queued', 'queue')}
            {cell('End to end', 'e2e')}
          </div>
          <div className="chart-wrap">
            <Chart marks={marks} times={h.map((s) => s.t)} height={90} fmt={(v) => fmtNum(v)}
              series={[{ label: 'First token ms', values: col(h, name, k('ttft'), 1000) }, { label: 'Between ms', values: col(h, name, k('itl'), 1000) }]} />
          </div>
        </>
      )
    },
  },
  {
    id: 'slo', title: 'Within targets', w: 2,
    options: [WINDOW, { key: 'warn', label: 'Warn below %', default: 95 }, { key: 'crit', label: 'Critical below %', default: 80 }],
    render: ({ marks, name, live, history, slo, setSlo, readonly }, o) => {
      const h = recent(history, o.window)
      const all = num(live, 'slo_combined')
      const tone = all == null ? '' : all * 100 < o.crit ? 'crit' : all * 100 < o.warn ? 'warn' : 'ok'
      const target = (m: string) => {
        const [v, u] = dur(slo?.[m] ?? null)
        return `under ${v} ${u ?? ''}`
      }
      return (
        <>
          <Hero label="Requests meeting every target" value={pct(all)} unit="%" tone={tone}
            sub={tone === 'crit' ? `▲ Critical: under ${o.crit}%` : tone === 'warn' ? `▲ Below ${o.warn}%` : undefined} />
          <div className="metrics">
            <Metric label="First token" value={pct(num(live, 'slo_ttft'))} unit="%" sub={target('ttft')} />
            <Metric label="Between tokens" value={pct(num(live, 'slo_itl'))} unit="%" sub={target('itl')} />
            <Metric label="Per output token" value={pct(num(live, 'slo_tpot'))} unit="%" sub={target('tpot')} />
            <Metric label="End to end" value={pct(num(live, 'slo_e2e'))} unit="%" sub={target('e2e')} />
          </div>
          {slo && !readonly && <details><summary>Change targets</summary><SloEditor key={JSON.stringify(slo)} slo={slo} setSlo={setSlo} /></details>}
          <div className="chart-wrap">
            <Chart marks={marks} times={h.map((s) => s.t)} height={90} max={100} fmt={(v) => `${fmtNum(v)}%`}
              series={[{ label: 'All targets', values: col(h, name, 'slo_combined', 100) }]} />
          </div>
        </>
      )
    },
  },
  {
    id: 'requests', title: 'Requests', w: 2, options: [WINDOW],
    render: ({ marks, name, live, history }, o) => {
      const h = recent(history, o.window)
      const cap = num(live, 'max_concurrency')
      return (
        <>
          <Hero label="Running now" value={fmtNum(num(live, 'running'))} sub={cap ? `room for about ${fmtNum(cap, 1)} at full context` : undefined} />
          <div className="metrics">
            <Metric label="Waiting" value={fmtNum(num(live, 'waiting'))} />
            <Metric label="Finished" value={fmtNum(num(live, 'requests_total'))} />
            <Metric label="Preempted" value={fmtNum(num(live, 'preemptions'))} sub="in the last second" />
          </div>
          <div className="chart-wrap">
            <Chart marks={marks} times={h.map((s) => s.t)} height={90} fmt={(v) => fmtNum(v)}
              series={[{ label: 'Running', values: col(h, name, 'running') }, { label: 'Waiting', values: col(h, name, 'waiting') }]} />
          </div>
        </>
      )
    },
  },
  {
    id: 'cache', title: 'Cache and drafting', w: 2, options: [WINDOW],
    render: ({ marks, name, live, history }, o) => {
      const h = recent(history, o.window)
      const pool = num(live, 'kv_pool_tokens')
      const drafting = num(live, 'draft_acceptance') != null || num(live, 'spec_drafted_total') != null
      return (
        <>
          <Hero label="KV cache in use" value={fmtNum(num(live, 'kv_used_pct'), 1)} unit="%" sub={pool ? `of ${fmtNum(pool)} tokens` : undefined} />
          <div className="metrics">
            <Metric label="Prefix cache hits" value={pct(num(live, 'prefix_hit_rate'))} unit="%" />
            {drafting ? (
              <>
                <Metric label="Draft tokens accepted" value={pct(num(live, 'draft_acceptance'))} unit="%" />
                <Metric label="Accepted per step" value={fmtNum(num(live, 'draft_accept_len'), 2)} unit="tok" />
                <Metric label="Accepted of drafted" value={fmtNum(num(live, 'spec_accepted_total'))} sub={`of ${fmtNum(num(live, 'spec_drafted_total'))}`} />
              </>
            ) : <Metric label="Speculative decoding" value="Off" />}
          </div>
          <div className="chart-wrap">
            <Chart marks={marks} times={h.map((s) => s.t)} height={90} max={100} fmt={(v) => `${fmtNum(v)}%`}
              series={[{ label: 'KV cache', values: col(h, name, 'kv_used_pct') }, ...(drafting ? [{ label: 'Accepted', values: col(h, name, 'draft_acceptance', 100) }] : [{ label: 'Prefix hits', values: col(h, name, 'prefix_hit_rate', 100) }])]} />
          </div>
        </>
      )
    },
  },
]

export const MODEL_DEFAULTS: TileConf[] = MODEL_TILES.map((t) => ({ id: t.id, w: t.w }))
