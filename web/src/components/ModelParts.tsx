import { useState } from 'react'
import { api } from '../api'

export type Slo = Record<string, number>

export function Hero({ label, value, unit, sub, tone, accent }: { label: string; value: string; unit?: string; sub?: string; tone?: string; accent?: boolean }) {
  return (
    <div className={`hero ${tone ? `tone-${tone}` : ''} ${accent ? 'accent' : ''}`}>
      <div className="label">{label}</div>
      <div className="value">{value}{unit && <small>{unit}</small>}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}

export function Metric({ label, value, unit, sub }: { label: string; value: string; unit?: string; sub?: string }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}{unit && <small>{unit}</small>}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}

const NAMES: Record<string, string> = { ttft: 'Time to first token', itl: 'Inter-token latency', tpot: 'Time per output token', e2e: 'End to end' }

export function SloEditor({ slo, setSlo }: { slo: Slo; setSlo: (s: Slo) => void }) {
  const [draft, setDraft] = useState(() => Object.fromEntries(Object.entries(slo).map(([k, v]) => [k, String(Math.round(v * 1000))])))
  const [err, setErr] = useState<string | null>(null)
  const save = () => {
    const body = Object.fromEntries(Object.entries(draft).map(([k, v]) => [k, Number(v) / 1000]))
    api<Slo>('/api/settings/slo', { method: 'PUT', json: body }).then((s) => { setSlo(s); setErr(null) }).catch((e: Error) => setErr(e.message))
  }
  return (
    <div className="slo-edit">
      {Object.keys(slo).map((k) => (
        <label key={k}>{NAMES[k] ?? k} (ms)
          <input type="number" min={1} value={draft[k] ?? ''} onChange={(e) => setDraft({ ...draft, [k]: e.target.value })} />
        </label>
      ))}
      <button onClick={save}>Save targets</button>
      {err && <span className="bad">{err}</span>}
    </div>
  )
}
