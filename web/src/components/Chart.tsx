import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { fmtTime } from '../api'

export type Series = { label: string; values: (number | null)[]; /** CSS variable for the line, instead of the next slot. */ color?: string }
/** Something that happened to the model (a start, a settings change), drawn as a line across the chart. */
export type Mark = { t: number; label: string }

type Props = {
  times: number[]
  series: Series[]
  /** Smallest plot height; the plot grows to fill its tile when the tile is taller. */
  height?: number
  min?: number
  max?: number
  fmt?: (v: number) => string
  /** Smallest top of the y scale, so an idle chart doesn't repeat one tick label. */
  floor?: number
  /** A sparkline: no axes, no legend, same hover readout. */
  bare?: boolean
  marks?: Mark[]
  /** Scale y to the data's own range instead of from zero, so small changes show. */
  fit?: boolean
  /** Give every series its own fitted y scale (sparklines only), so two rates of different size both show their shape. */
  split?: boolean
}

// Categorical slots, fixed order (validated for the dark surface; see notes/dgx-kit-design.md).
const SLOTS = ['--s1', '--s2', '--s3', '--s4', '--s5']

const clock = (t: number) => fmtTime(t, true)

/** Dashed lines at restarts and settings changes, labelled at the top unless the chart is a sparkline. */
function markers(get: () => Mark[] | undefined, color: string, text: string, font: string, labels: boolean): uPlot.Plugin {
  return {
    hooks: {
      draw: (u) => {
        const ms = get()
        if (!ms?.length) return
        const [lo, hi] = [u.scales.x.min ?? 0, u.scales.x.max ?? 0]
        const c = u.ctx
        const r = devicePixelRatio || 1
        c.save()
        c.strokeStyle = color
        c.lineWidth = r
        c.setLineDash([3 * r, 3 * r])
        c.font = font.replace(/(\d+)px/, (_m, n) => `${Number(n) * r}px`)
        c.fillStyle = text
        let lastRight = -Infinity
        for (const m of ms) {
          if (m.t < lo || m.t > hi) continue
          const x = Math.round(u.valToPos(m.t, 'x', true))
          c.beginPath()
          c.moveTo(x, u.bbox.top)
          c.lineTo(x, u.bbox.top + u.bbox.height)
          c.stroke()
          if (!labels) continue
          const w = c.measureText(m.label).width
          const left = x + 4 * r
          if (left + w > u.bbox.left + u.bbox.width) continue
          if (left < lastRight) continue
          c.fillText(m.label, left, u.bbox.top + 10 * r)
          lastRight = left + w + 8 * r
        }
        c.restore()
      },
    },
  }
}

/** Crosshair plus one readout listing every series at the hovered time. */
function readout(fmt: (v: number) => string, labels: string[], colors: string[]): uPlot.Plugin {
  let tip: HTMLDivElement
  return {
    hooks: {
      init: (u) => {
        tip = document.createElement('div')
        tip.className = 'chart-tip'
        tip.style.display = 'none'
        u.over.appendChild(tip)
        u.over.addEventListener('mouseleave', () => { tip.style.display = 'none' })
      },
      setCursor: (u) => {
        const i = u.cursor.idx
        if (i == null || u.cursor.left == null || u.cursor.left < 0) {
          tip.style.display = 'none'
          return
        }
        tip.replaceChildren()
        const time = document.createElement('div')
        time.className = 'when'
        time.textContent = clock(u.data[0][i])
        tip.appendChild(time)
        labels.forEach((label, s) => {
          const v = u.data[s + 1][i]
          const row = document.createElement('div')
          const key = document.createElement('i')
          key.style.background = colors[s]
          const val = document.createElement('b')
          val.textContent = v == null ? '–' : fmt(v)
          row.append(key, val)
          if (labels.length > 1) row.append(document.createTextNode(label))
          tip.appendChild(row)
        })
        tip.style.display = 'block'
        const w = u.over.clientWidth
        const x = u.cursor.left
        tip.style.left = `${x > w / 2 ? x - tip.offsetWidth - 10 : x + 10}px`
        tip.style.top = '4px'
      },
    },
  }
}

export function Chart({ times, series, height = 90, min = 0, max, fmt, floor = 1, bare = false, marks, fit = false, split = false }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)
  const fmtRef = useRef(fmt)
  const marksRef = useRef(marks)
  const labels = series.map((s) => s.label + (s.color ? '~' + s.color : '')).join('|')

  useEffect(() => {
    fmtRef.current = fmt
  })
  const markKey = (marks ?? []).map((m) => m.t).join(',')
  useEffect(() => {
    marksRef.current = marks
    plot.current?.redraw(false)
  }, [markKey]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const el = box.current
    if (!el) return
    const css = getComputedStyle(el)
    const tok = (n: string, d: string) => css.getPropertyValue(n).trim() || d
    const muted = tok('--muted', '#999')
    const grid = tok('--grid', '#2a2a2a')
    const names = labels.split('|').map((n) => n.split('~')[0])
    const colors = labels.split('|').map((n, i) => tok(n.split('~')[1] || SLOTS[i % SLOTS.length], '#3987e5'))
    const font = `11px ${tok('--font', 'system-ui')}`
    const f = (v: number) => (fmtRef.current ? fmtRef.current(v) : v.toLocaleString(undefined, { maximumFractionDigits: 1 }))
    const yRange = (_u: uPlot, lo: number | null, hi: number | null): uPlot.Range.MinMax => {
      if ((fit || split) && lo != null && hi != null) {
        const pad = Math.max((hi - lo) * 0.15, Math.abs(hi) * 0.02, 1e-6)
        return [Math.max(min, lo - pad), hi + pad]
      }
      return [Math.min(min, lo ?? 0), max ?? Math.max(hi ?? 0, floor) * 1.08]
    }
    // Split sparklines stack each series in its own lane, top to bottom, so the lines never cross.
    const lane = (i: number, n: number) => (u: uPlot, lo: number | null, hi: number | null): uPlot.Range.MinMax => {
      const [a, b] = yRange(u, lo, hi) as [number, number]
      const r = (b - a) * 1.25
      const top = b + i * r + 0.125 * (b - a)
      return [top - n * r, top]
    }
    const size = () => ({ width: el.clientWidth || 300, height: Math.max(el.clientHeight, 30) })
    const opts: uPlot.Options = {
      ...size(),
      legend: { show: false },
      padding: bare ? [2, 0, 2, 0] : [6, 4, 0, 0],
      cursor: { points: { show: false }, drag: { x: false, y: false }, y: false },
      scales: { x: { time: true }, ...(split
        ? Object.fromEntries(names.map((_, i) => [`y${i}`, { range: lane(i, names.length) }]))
        : { y: { range: yRange } }) },
      axes: bare ? [{ show: false }, { show: false }] : [
        { stroke: muted, grid: { show: false }, ticks: { show: false }, size: 22, font, space: 110,
          // uPlot's own labels are 12-hour ("3pm"); show the same 24-hour clock as everywhere else, with the day when ticks are a day apart
          values: (_u, vals, _axis, _space, incr) => vals.map((v) => (v == null ? '' : incr >= 86400
            ? new Date(v * 1000).toLocaleDateString([], { month: 'short', day: 'numeric' }) : fmtTime(v, incr < 60))) },
        { stroke: muted, grid: { stroke: grid, width: 1 }, ticks: { show: false }, size: 48, font, space: 30, values: (_u, vals) => vals.map((v) => (v == null ? '' : f(v))) },
      ],
      series: [{}, ...colors.map((c, i) => ({ scale: split ? `y${i}` : 'y', stroke: c, width: split ? 1.5 : 2, fill: names.length === 1 ? c + '1a' : undefined, points: { show: false } }))],
      plugins: [readout(f, names, colors), markers(() => marksRef.current, tok('--faint', '#666'), muted, font, !bare)],
    }
    plot.current = new uPlot(opts, [[], ...colors.map(() => [])] as unknown as uPlot.AlignedData, el)
    const ro = new ResizeObserver(() => plot.current?.setSize(size()))
    ro.observe(el)
    return () => {
      ro.disconnect()
      plot.current?.destroy()
      plot.current = null
    }
  }, [labels, min, max, floor, bare, fit, split])

  useEffect(() => {
    plot.current?.setData([times, ...series.map((s) => s.values)] as uPlot.AlignedData)
  }, [times, series])

  return (
    <div className="chart">
      {series.length > 1 && !bare && (
        <div className="legend-row">
          {series.map((s, i) => <span key={s.label}><i style={{ background: `var(${SLOTS[i % SLOTS.length]})` }} />{s.label}</span>)}
        </div>
      )}
      <div className="plot" style={{ minHeight: height }}>
        <div ref={box} className="plot-in" />
      </div>
    </div>
  )
}

/** One series, no axes: a sparkline with the same hover readout. */
export function Spark({ times, values, max, height = 36, label = 'Value', fmt, marks }: { times: number[]; values: (number | null)[]; max?: number; height?: number; label?: string; fmt?: (v: number) => string; marks?: Mark[] }) {
  return <Chart bare fit={max == null} times={times} series={[{ label, values }]} max={max} height={height} fmt={fmt} marks={marks} />
}

/** Several rates on one sparkline, each on its own fitted scale; the caller shows the legend and current values. */
export function MultiSpark({ times, series, height = 36, fmt }: { times: number[]; series: Series[]; height?: number; fmt?: (v: number) => string }) {
  return <Chart bare split times={times} series={series} height={height} fmt={fmt} />
}

