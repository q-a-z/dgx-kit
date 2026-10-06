import { Modal } from '../components/Confirm'
import type { Snapshot } from '../api'
import { PRICES, PRICES_AS_OF } from '../pricing'

const usd = (n: number) => `$${n < 100 ? n.toFixed(2) : Math.round(n).toLocaleString('en-US')}`
const tok = (n: number) => n >= 1e9 ? `${(n / 1e9).toFixed(2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} K` : `${Math.round(n)}`

/** What the tokens served since the models started would have cost at Claude and OpenAI list prices. */
export function CostModal({ open, onClose, latest }: { open: boolean; onClose: () => void; latest: Snapshot | null }) {
  const up = Object.values(latest?.models ?? {}).filter((x) => x.up)
  const total = (k: string) => up.reduce((a, x) => a + (typeof x[k] === 'number' ? (x[k] as number) : 0), 0)
  const tin = total('prompt_tokens_total')
  const tout = total('gen_tokens_total')
  return (
    <Modal open={open} onClose={onClose} title="What this would cost elsewhere">
      <div className="modal-body">
        <p className="muted small">{tok(tin)} tokens in · {tok(tout)} tokens out since the models started.</p>
        <table className="cost">
          <thead><tr><th>Model</th><th>In</th><th>Out</th><th>Total</th></tr></thead>
          <tbody>
            {PRICES.map((p) => {
              const i = (tin / 1e6) * p.in, o = (tout / 1e6) * p.out
              return <tr key={p.model}><td>{p.vendor} {p.model}<small className="muted"> ${p.in} / ${p.out}</small></td><td>{usd(i)}</td><td>{usd(o)}</td><td><b>{usd(i + o)}</b></td></tr>
            })}
          </tbody>
        </table>
        <p className="muted small">List prices per million tokens (in / out) as of {PRICES_AS_OF}, without caching or batch discounts. Counters restart with each model, so this is the cost of the current runs only.</p>
      </div>
      <div className="row modal-actions"><button autoFocus onClick={onClose}>Close</button></div>
    </Modal>
  )
}
