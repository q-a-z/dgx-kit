/** A horizontal tape gauge: the fill is the reading, the marks are the warning and critical limits (0 = off). */
export function Tape({ value, min = 0, max, warn, crit }: { value: number | null | undefined; min?: number; max: number; warn?: number; crit?: number }) {
  const pos = (v: number) => `${Math.min(100, Math.max(0, ((v - min) / (max - min || 1)) * 100))}%`
  const lvl = level(value, warn, crit)
  return (
    <div className={`tape ${lvl}`} role="meter" aria-valuemin={min} aria-valuemax={max} aria-valuenow={value ?? undefined}>
      {value != null && <div className="fill" style={{ width: pos(value) }} />}
      {warn ? <div className="mark warn" style={{ left: pos(warn) }} /> : null}
      {crit ? <div className="mark crit" style={{ left: pos(crit) }} /> : null}
    </div>
  )
}

function level(v: number | null | undefined, warn?: number, crit?: number) {
  return v == null ? '' : crit && v >= crit ? 'crit' : warn && v >= warn ? 'warn' : ''
}

/** Status is never colour alone: a symbol and words go with it. */
export function Status({ level, warn, crit, unit }: { level: string; warn: number; crit: number; unit: string }) {
  if (!level) return null
  return <span className={`status ${level}`}>{level === 'crit' ? '▲ Critical' : '▲ High'}<span className="muted"> over {level === 'crit' ? crit : warn} {unit}</span></span>
}

