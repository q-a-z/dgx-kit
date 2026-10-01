import { useState } from 'react'
import { api, fmtBytes, fmtNum, type Plan } from '../api'
import { usePoll } from '../usePoll'
import { select } from '../nav'
import { toast } from '../toast'

type Item = {
  name: string
  path: string
  format: string
  gguf_files: string[]
  architecture: string | null
  quantization: string | null
  size_bytes: number
  kind: 'model' | 'draft'
  kind_set_by_user?: boolean
  used_by?: string | null
}
export type Lib = { paths: string[]; missing: string[]; models: Item[]; drafts: Item[]; unreadable?: string[]; incomplete?: string[] }

export function Folders({ lib, onSaved }: { lib: Pick<Lib, 'paths' | 'missing'>; onSaved: () => void }) {
  const [text, setText] = useState(lib.paths.join('\n'))
  const [err, setErr] = useState<string | null>(null)
  const save = () =>
    api('/api/library/paths', { method: 'PUT', json: { paths: text.split('\n') } })
      .then(() => { setErr(null); toast('Model folders saved.'); onSaved() })
      .catch((e: Error) => setErr(e.message))
  return (
    <section className="card wide">
      <h2>Model folders</h2>
      <p className="muted">DGX-kit looks for models in these folders, one per line: model folders with config.json, GGUF files and Hugging Face caches. It only reads them.</p>
      <textarea rows={Math.max(2, lib.paths.length + 1)} value={text} onChange={(e) => setText(e.target.value)} />
      <div className="row" style={{ marginTop: 8 }}>
        <button className="primary" onClick={save} disabled={text.trim() === lib.paths.join('\n')}>Save folders</button>
        {lib.missing.length > 0 && <span className="bad">Not found: {lib.missing.join(', ')}</span>}
      </div>
      {err && <p className="bad">{err}</p>}
    </section>
  )
}

export type { Item }

function Row({ item, drafts, onChange }: { item: Item; drafts: Item[]; onChange: (msg: string, setUp?: string) => void }) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState('')
  const [gguf, setGguf] = useState(item.gguf_files[0] ?? '')
  const [name, setName] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const other = item.kind === 'model' ? 'draft' : 'model'
  const setKind = () => api('/api/library/kind', { method: 'PUT', json: { path: item.path, kind: other } }).then(() => onChange(`Marked ${item.name} as a ${other}.`))
  const setup = () =>
    api<{ recipe: { name: string }; plan: Plan | null }>('/api/library/setup', {
      method: 'POST',
      json: { path: item.path, draft_path: draft || null, gguf_file: gguf || null, name: name.trim() || null },
    })
      .then((r) => onChange(`Set up ${r.recipe.name}. ${r.plan ? (r.plan.fits ? `It fits with ${fmtNum(r.plan.context_tokens)} context.` : `It doesn't fit now: ${r.plan.reason}.`) : ''} Opening it now.`, r.recipe.name))
      .catch((e: Error) => setErr(e.message))
  return (
    <li className={`disk-row ${open ? 'open' : ''}`}>
      <div className="dn" title={item.path}>
        <strong>{item.path.includes('/models--') ? item.name.split('/').pop() : item.name}</strong>
        <span className="muted">{shortPath(item.path)}</span>
      </div>
      <span className="dt muted" title={[item.architecture, item.quantization].filter(Boolean).join(' · ')}>{typeLabel(item)}</span>
      <span className="ds">{fmtBytes(item.size_bytes)}</span>
      <span className="da">
        {item.kind === 'model' && (item.used_by
          ? <a className="in-use" href={`#/model/${encodeURIComponent(item.used_by)}`} title={`Set up as ${item.used_by}`}>✓ Set up</a>
          : !open && <button onClick={() => setOpen(true)}>Set up…</button>)}
        <button className="link" title={item.kind_set_by_user ? 'You chose this' : 'Detected automatically'} onClick={setKind}>Mark as {other}</button>
      </span>
      {open && (
        <div className="pick-setup">
          <label>Name<input placeholder="automatic" value={name} onChange={(e) => setName(e.target.value)} /></label>
          <label>Draft
            <select value={draft} onChange={(e) => setDraft(e.target.value)}>
              <option value="">No draft</option>
              {drafts.map((d) => <option key={d.path} value={d.path}>{d.name}</option>)}
            </select>
          </label>
          {item.gguf_files.length > 1 && (
            <label>GGUF file
              <select value={gguf} onChange={(e) => setGguf(e.target.value)}>{item.gguf_files.map((f) => <option key={f}>{f}</option>)}</select>
            </label>
          )}
          <span className="grow" />
          <button onClick={() => setOpen(false)}>Cancel</button>
          <button className="primary" onClick={setup}>Set up</button>
          {err && <p className="bad">{err}</p>}
        </div>
      )}
    </li>
  )
}

/** "Qwen3MoeForCausalLM · modelopt" reads as "Qwen3Moe · modelopt". */
function typeLabel(i: Item) {
  const arch = (i.architecture ?? i.format).replace(/For(CausalLM|ConditionalGeneration|Causal)$|Model$/, '')
  const q = i.quantization === 'compressed-tensors' ? 'compressed' : i.quantization
  return [arch, q && q !== arch.toLowerCase() ? q : null].filter(Boolean).join(' · ')
}

/** Models already in a folder on this box, ready to set up without downloading. */
export function LocalModels({ onSetUp }: { onSetUp: (name: string) => void }) {
  const { data, reload } = usePoll<Lib>('/api/library', 15000)
  const [note, setNote] = useState<string | null>(null)
  if (!data) return <p className="muted">Looking through the model folders…</p>
  const done = (msg: string, setUp?: string) => { if (setUp) return onSetUp(setUp); setNote(msg); reload() }
  const table = (items: Item[], empty: string) =>
    items.length === 0 ? <p className="muted">{empty}</p> : (
      <ul className="disk-list">{items.map((i) => <Row key={i.path} item={i} drafts={data.drafts} onChange={done} />)}</ul>
    )
  return (
    <>
      <p className="muted">
        Found in {data.paths.length ? data.paths.join(', ') : 'no folders'}. <a href="#/settings">Change folders</a>
        {data.missing.length > 0 && <span className="bad"> Not found: {data.missing.join(', ')}</span>}
      </p>
      {note && <div className="note">{note}</div>}
      {!!data.unreadable?.length && <p className="bad">Couldn’t read {data.unreadable.join(', ')} (permissions), so models there aren’t listed.</p>}
      {!!data.incomplete?.length && (
        <details className="partial"><summary className="muted">{data.incomplete.length} folders have a config but no weights (unfinished downloads)</summary>
          <ul>{data.incomplete.map((p) => <li key={p} title={p}>{shortPath(p)}</li>)}</ul>
        </details>
      )}
      <h4 className="sub">Models ({data.models.length})</h4>
      {table(data.models, 'No models found in these folders.')}
      <h4 className="sub">Drafts for speculative decoding ({data.drafts.length})</h4>
      {table(data.drafts, 'No draft models found in these folders.')}
    </>
  )
}

export function useLibrary() {
  return usePoll<Lib>('/api/library', 15000)
}

/** Every model and draft in the model folders, with Set up beside each. */
export function LibraryPage() {
  return (
    <div className="page narrow-page">
      <h1>Models on disk</h1>
      <p className="muted lead">Weights already on this box. Set one up to run it from the dashboard; the files stay where they are.</p>
      <LocalModels onSetUp={(name) => select(name)} />
    </div>
  )
}

/** A Hugging Face cache snapshot reads as its repo id; anything else as its path under home. */
function shortPath(p: string) {
  const hf = p.match(/models--([^/]+?)--([^/]+)\/snapshots\//)
  return hf ? `Hugging Face cache · ${hf[1]}/${hf[2]}` : p.replace(/^\/home\/[^/]+/, '~')
}
