import { useState } from 'react'
import { api, fmtBytes, type Plan, type Recipe } from '../api'
import { Icon } from '../components/Icon'
import { prettyName } from '../fleet'
import { select } from '../nav'
import { usePoll } from '../usePoll'
import type { Item, Lib } from './Library'
import { AddModel } from './Models'

export type Ghost = { name: string; bytes: number; fits: boolean } | null

/** Bring a model onto the box, in the same panel a model's details use. What it would take shows in the free space. */
export function RunPanel({ readonly, onGhost, onChanged }: { readonly: boolean; onGhost: (g: Ghost) => void; onChanged: () => void }) {
  const lib = usePoll<Lib>('/api/library', 30000)
  const [from, setFrom] = useState<'box' | 'hf' | null>(null)
  const local = (lib.data?.models ?? []).filter((m) => !m.used_by)
  const src = from ?? (local.length ? 'box' : 'hf')
  const found = (r: Recipe | null, p: Plan | null) =>
    onGhost(r ? { name: r.name, bytes: Math.round((r.weights_bytes + r.draft_weights_bytes) * 1.1) + (p?.kv_bytes ?? 0), fits: p?.fits ?? true } : null)
  const added = (name: string) => { onGhost(null); onChanged(); select(name) }
  return (
    <section className="panel" aria-label="Run a model">
      <div className="panel-head">
        <h2 className="panel-name">Run a model</h2>
        <div className="seg" role="tablist">
          <button role="tab" aria-selected={src === 'box'} className={src === 'box' ? 'active' : ''} onClick={() => setFrom('box')}>On this box{lib.data ? ` (${local.length})` : ''}</button>
          <button role="tab" aria-selected={src === 'hf'} className={src === 'hf' ? 'active' : ''} onClick={() => setFrom('hf')}>From Hugging Face</button>
        </div>
        <span className="grow" />
        <button className="ic" title="Close" aria-label="Close" onClick={() => { onGhost(null); select(undefined) }}><Icon name="close" /></button>
      </div>
      {readonly && <p className="advice warn">Read-only: you can look models up, but nothing will download or start on this box.</p>}
      {src === 'hf' ? (
        <>
          <p className="muted">Paste a model id. You’ll see its size and where it would sit in memory before anything downloads.</p>
          <AddModel onAdded={added} onFound={found} />
        </>
      ) : !lib.data ? <p className={lib.error ? 'bad' : 'muted'}>{lib.error ? `Couldn’t read the model folders: ${lib.error}` : 'Reading the model folders…'}</p>
        : local.length === 0 ? <p className="muted">No models in the model folders that aren’t set up already. <a href="#/library">Models on disk</a></p>
        : (
          <>
            <ul className="pick">{local.map((m) => <LocalRow key={m.path} item={m} drafts={lib.data!.drafts} onAdded={added} />)}</ul>
            <p className="muted small"><a href="#/library">Models on disk</a> lists drafts too, and lets you fix a model marked as a draft.</p>
          </>
        )}
    </section>
  )
}

/** Drafts made for this model first: the longer the shared start of the names, the likelier. */
function rank(model: string, drafts: Item[]) {
  const norm = (n: string) => n.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  const a = norm(model)
  const shared = (b: string) => { let i = 0; while (i < a.length && a[i] === b[i]) i++; return i }
  return [...drafts].map((d) => ({ d, n: shared(norm(d.name)) })).sort((x, y) => y.n - x.n).map((x) => ({ ...x.d, match: x.n >= 4 }))
}

function LocalRow({ item, drafts, onAdded }: { item: Item; drafts: Item[]; onAdded: (name: string) => void }) {
  const [open, setOpen] = useState(false)
  const ranked = rank(item.name, drafts)
  const [draft, setDraft] = useState(ranked.find((d) => d.match)?.path ?? '')
  const [gguf, setGguf] = useState(item.gguf_files[0] ?? '')
  const [err, setErr] = useState<string | null>(null)
  const setup = () =>
    api<{ recipe: { name: string } }>('/api/library/setup', { method: 'POST', json: { path: item.path, draft_path: draft || null, gguf_file: gguf || null } })
      .then((r) => onAdded(r.recipe.name)).catch((e: Error) => setErr(e.message))
  return (
    <li className={open ? 'open' : ''}>
      <div className="pick-row">
        <strong title={item.path}>{prettyName({ name: item.name, row: { repo: item.name } } as never)}</strong>
        <span className="muted">{[item.quantization, fmtBytes(item.size_bytes)].filter(Boolean).join(' · ')}</span>
        <span className="grow" />
        {!open && <button onClick={() => setOpen(true)}>Run…</button>}
      </div>
      {open && (
        <div className="pick-setup">
          <label>Draft
            <select value={draft} onChange={(e) => setDraft(e.target.value)}>
              <option value="">No draft</option>
              {ranked.map((d) => <option key={d.path} value={d.path}>{d.name}{d.match ? ' (matches)' : ''}</option>)}
            </select>
          </label>
          {item.gguf_files.length > 1 && (
            <label>File
              <select value={gguf} onChange={(e) => setGguf(e.target.value)}>{item.gguf_files.map((f) => <option key={f}>{f}</option>)}</select>
            </label>
          )}
          <span className="grow" />
          <button onClick={() => setOpen(false)}>Cancel</button>
          <button className="primary" onClick={setup}>Set up and open</button>
          {err && <p className="bad">{err}</p>}
        </div>
      )}
    </li>
  )
}
