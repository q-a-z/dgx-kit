import { Modal } from '../components/Confirm'
import { fmtTime, resetStats, type Snapshot } from '../api'
import { toast } from '../toast'
import { CACHE_SHARES, PRICES, PRICES_AS_OF } from '../pricing'

const usd = (n: number) => `$${n < 100 ? n.toFixed(2) : Math.round(n).toLocaleString('en-US')}`
const tok = (n: number) => n >= 1e9 ? `${(n / 1e9).toFixed(2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} K` : `${Math.round(n)}`

/** What the tokens served since the models started would have cost at Claude and OpenAI list prices. */
export function CostModal({ open, onClose, latest }: { open: boolean; onClose: () => void; latest: Snapshot | null }) {
  const up = Object.values(latest?.models ?? {}).filter((x) => x.up)
  const total = (k: string) => up.reduce((a, x) => a + (typeof x[k] === 'number' ? (x[k] as number) : 0), 0)
  const tin = total('prompt_tokens_total')
  const tout = total('gen_tokens_total')
  const since = Math.max(0, ...up.map((x) => (typeof x.stats_since === 'number' ? x.stats_since : 0)))
  return (
    <Modal open={open} onClose={onClose} title="What this would cost elsewhere">
      <div className="modal-body">
        <p className="muted small">{tok(tin)} tokens in · {tok(tout)} tokens out since the models started{since ? ` or were reset (${fmtTime(since)})` : ''}.</p>
        <table className="cost">
          <thead><tr><th>Model</th><th>In</th><th>Out</th><th>Total</th>{CACHE_SHARES.map((s) => <th key={s} title="Share of input read from the provider's cache">{s * 100}% cached</th>)}</tr></thead>
          <tbody>
            {PRICES.map((p) => {
              const i = (tin / 1e6) * p.in, o = (tout / 1e6) * p.out
              const withCache = (s: number) => (tin / 1e6) * (p.in * (1 - s) + p.cached * s) + o
              return <tr key={p.model}><td>{p.vendor} {p.model}<small className="muted"> ${p.in} / ${p.out}</small></td><td>{usd(i)}</td><td>{usd(o)}</td><td><b>{usd(i + o)}</b></td>{CACHE_SHARES.map((s) => <td key={s}><b>{usd(withCache(s))}</b></td>)}</tr>
            })}
          </tbody>
        </table>
        <p className="muted small">List prices per million tokens (in / out) as of {PRICES_AS_OF}, no batch discounts. The first Total pays full price for every input token; the cached columns assume 50% and 90% of input is read from the provider's cache at its cache-read price. Counters restart with each model and with Reset stats, so this is the cost since then only.</p>
      </div>
      <div className="row modal-actions">
        <button onClick={() => resetStats().then(() => toast('Stats reset.')).catch((e: Error) => toast(e.message, true))}
          title="Start the token and request totals from zero">Reset stats</button>
        <button autoFocus onClick={onClose}>Close</button>
      </div>
    </Modal>
  )
}
