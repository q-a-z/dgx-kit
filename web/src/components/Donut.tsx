import { Icon } from './Icon'

/** A 270° ring gauge: faint colour zones on the track, the value arc in the colour of the zone it's in.
 *  With `second`, two things share one gauge: the outer ring is `value`, the inner ring is `second.value`. */
export type Zone = { upTo: number; color: string }

const R = 36
const R_SHARED = 38, R2 = 29 // outer and inner ring of a shared gauge; the middle stays wide enough for two readings
const C = 2 * Math.PI * R
const SWEEP = 0.75 // of the circle

export function Donut({ value, min = 0, max, unit, label, zones, color = 'var(--s1)', size = 112, tag, second, curve, fine }: {
  value: number | null | undefined; min?: number; max: number; unit: string; label: string; zones?: Zone[]; color?: string; size?: number
  tag?: string; second?: { value: number | null | undefined; tag: string }
  /** Square-root scale, so a trickle still moves the ring on a gauge that goes up to a flood. */
  curve?: boolean
  /** One decimal under 10, for small rates. */
  fine?: boolean
}) {
  const frac = (v: number) => { const f = Math.min(1, Math.max(0, (v - min) / (max - min))); return curve ? Math.sqrt(f) : f }
  const zoneOf = (v: number) => zones?.find((z) => v <= z.upTo)?.color ?? zones?.[zones.length - 1]?.color ?? color
  const arc = (from: number, to: number) => ({ strokeDasharray: `${(to - from) * SWEEP * C} ${C}`, strokeDashoffset: -from * SWEEP * C })
  let prev = min
  const ring = (v: number | null | undefined, r: number, w: number) => (
    <>
      <circle cx="48" cy="48" r={r} stroke="var(--track)" strokeWidth={w} style={arc(0, 1)} />
      {v != null && <circle cx="48" cy="48" r={r} stroke={zoneOf(v)} strokeWidth={w} strokeLinecap="round" style={{ ...arc(0, Math.max(frac(v), 0.005)), transition: 'stroke-dasharray .5s ease' }} />}
    </>
  )
  const num = (v: number | null | undefined) => (v == null ? '–' : fine && v < 10 ? v.toFixed(1) : Math.round(v))
  return (
    <div className="donut" role="meter" aria-label={label} aria-valuemin={min} aria-valuemax={max} aria-valuenow={value ?? undefined} aria-valuetext={second ? `${tag} ${num(value)}${unit}, ${second.tag} ${num(second.value)}${unit}` : undefined}>
      <svg viewBox="0 0 96 96" width={size} height={size}>
        <g transform="rotate(135 48 48)" fill="none">
          {second ? ring(value, R_SHARED, 5) : <circle cx="48" cy="48" r={R} stroke="var(--track)" strokeWidth="8" style={arc(0, 1)} />}
          {zones?.map((z) => {
            const a = frac(prev), b = frac(Math.min(z.upTo, max))
            prev = z.upTo
            return <circle key={z.upTo} cx="48" cy="48" r={(second ? R_SHARED : R) + 7} stroke={z.color} strokeOpacity={0.55} strokeWidth="2" style={arc(a, b)} />
          })}
          {second ? ring(second.value, R2, 5)
            : value != null && <circle cx="48" cy="48" r={R} stroke={zoneOf(value)} strokeWidth="8" strokeLinecap="round" style={{ ...arc(0, Math.max(frac(value), 0.005)), transition: 'stroke-dasharray .5s ease' }} />}
        </g>
        {second ? (
          <>
            <g style={{ color: 'var(--muted)' }} transform="translate(30 33.5)"><Icon name={(tag ?? '').toLowerCase()} size={11} /></g>
            <text x="44" y="44" className="dv dv-sm">{num(value)}</text>
            <g style={{ color: 'var(--muted)' }} transform="translate(30 47.5)"><Icon name={second.tag.toLowerCase()} size={11} /></g>
            <text x="44" y="58" className="dv dv-sm">{num(second.value)}</text>
            <text x="48" y="71" textAnchor="middle" className="du">{unit}</text>
          </>
        ) : (
          <>
            <text x="48" y="50" textAnchor="middle" className="dv">{num(value)}</text>
            <text x="48" y="66" textAnchor="middle" className="du">{unit}</text>
          </>
        )}
      </svg>
      <span className="lbl">{label}</span>
    </div>
  )
}
