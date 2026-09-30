import { useState } from 'react'
import { api, fmtBytes } from '../api'
import { Modal } from '../components/Confirm'
import { usePoll } from '../usePoll'
import type { Item, Lib } from './Library'
import { Pending } from './Settings'

type Preview = { path: string; name: string; folder: string; size_bytes: number; used_by: string[]; also_removes: string[] }

/** The only place files leave the disk: four confirmations, and the last one is typing the folder's name. */
export function DiskDelete() {
  const lib = usePoll<Lib>('/api/library', 30000)
  const [target, setTarget] = useState<Item | null>(null)
  const [pre, setPre] = useState<Preview | null>(null)
  const [step, setStep] = useState(1)
  const [typed, setTyped] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)

  const open = (item: Item) => {
    setTarget(item); setPre(null); setStep(1); setTyped(''); setErr(null); setNote(null)
    api<Preview>(`/api/library/delete-preview?path=${encodeURIComponent(item.path)}`).then(setPre).catch((e: Error) => setErr(e.message))
  }
  const close = () => setTarget(null)
  const remove = () => pre && api<{ freed_bytes: number }>('/api/library/delete', { method: 'POST', json: { path: pre.path, confirm: typed } })
    .then((r) => { setNote(`Deleted ${pre.name} from disk, ${fmtBytes(r.freed_bytes)} freed.`); close(); lib.reload() })
    .catch((e: Error) => setErr(e.message))

  const items = [...(lib.data?.models ?? []), ...(lib.data?.drafts ?? [])]
  const size = pre ? fmtBytes(pre.size_bytes) : '…'
  return (
    <section className="card wide">
      {!lib.data ? <Pending error={lib.error} what="the models on disk" /> : items.length === 0 ? <p className="muted">No model folders found.</p> : (
        <table className="disk-table"><tbody>
          {items.map((i) => (
            <tr key={i.path}>
              <td><strong>{i.name}</strong><div className="muted small">{i.path}</div></td>
              <td className="muted">{fmtBytes(i.size_bytes)}</td>
              <td className="muted small">{i.used_by ? `used by ${i.used_by}` : ''}</td>
              <td><button className="danger" onClick={() => open(i)}>Delete from disk…</button></td>
            </tr>
          ))}
        </tbody></table>
      )}
      {note && <p className="note">{note}</p>}
      <Modal open={!!target} onClose={close} title={pre && step <= 4 ? `Delete ${pre.name} from disk` : 'Delete from disk'}>
        {err && <><div className="modal-body bad">{err}</div><div className="row modal-actions"><button autoFocus onClick={close}>Close</button></div></>}
        {!err && !pre && <div className="modal-body">Checking…</div>}
        {!err && pre && (
          <>
            <p className="muted small">Step {step} of 4</p>
            <div className="modal-body">
              {step === 1 && <>This permanently deletes <b>{size}</b> from <code>{pre.path}</code>. It does not go to a trash.</>}
              {step === 2 && <>
                {pre.used_by.length ? <>DGX-kit models that use these files: <b>{pre.used_by.join(', ')}</b>. They will show as not downloaded.</> : 'No DGX-kit model uses these files.'}
                {pre.also_removes.length > 0 && <> Everything inside goes too, including <b>{pre.also_removes.join(', ')}</b>.</>}
              </>}
              {step === 3 && <>There is no undo. If you don’t have these weights somewhere else, they are gone for good, and getting them back means downloading again.</>}
              {step === 4 && <>
                Type the folder name to confirm: <code>{pre.folder}</code>
                <input autoFocus style={{ display: 'block', marginTop: 8, width: '100%' }} value={typed} onChange={(e) => setTyped(e.target.value)} />
              </>}
            </div>
            <div className="row modal-actions">
              <button autoFocus={step < 4} onClick={close}>Cancel</button>
              {step < 4
                ? <button className="danger" onClick={() => setStep(step + 1)}>{step === 3 ? 'Yes, keep going' : 'Continue'}</button>
                : <button className="danger" disabled={typed !== pre.folder} onClick={remove}>Delete forever</button>}
            </div>
          </>
        )}
      </Modal>
    </section>
  )
}
