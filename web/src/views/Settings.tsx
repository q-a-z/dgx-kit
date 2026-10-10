import { useEffect, useState } from 'react'
import { api } from '../api'
import { ChangePassword } from '../Login'
import { usePoll } from '../usePoll'
import { Folders } from './Library'
import { DiskDelete } from './DiskDelete'
import { GatewayCard } from './Models'
import { go } from '../nav'
import { toast } from '../toast'
import { Images, Log } from './Other'
import { SystemInfo } from './SystemInfo'

type Lib = Parameters<typeof Folders>[0]['lib']

const TABS = [
  { id: 'system', label: 'System' },
  { id: 'gateway', label: 'Gateway' },
  { id: 'slots', label: 'GPU slots' },
  { id: 'hf', label: 'Hugging Face' },
  { id: 'images', label: 'Engine images' },
  { id: 'folders', label: 'Model folders' },
  { id: 'import', label: 'Import' },
  { id: 'disk', label: 'Delete from disk' },
  { id: 'activity', label: 'Activity' },
  { id: 'password', label: 'Password' },
]

/** Set-up-once things, one tab each: how clients reach the models, which engine images run them, where models live.
 *  The tab is in the address (#/settings/images), so a reload, Back and a shared link land on the same one. */
export function Settings({ canChangePassword, tab }: { canChangePassword: boolean; tab?: string }) {
  const lib = usePoll<Pick<Lib, 'paths' | 'missing'>>('/api/library/paths', 30000)
  const tabs = TABS.filter((t) => t.id !== 'password' || canChangePassword)
  const current = tabs.find((t) => t.id === tab)?.id ?? 'system'
  return (
    <div className="page narrow-page settings">
      <h1>Settings</h1>
      <div className="tabs" role="tablist" aria-label="Settings">
        {tabs.map((t) => (
          <button key={t.id} role="tab" aria-selected={current === t.id} className={current === t.id ? 'on' : ''} onClick={() => go({ page: 'settings', arg: t.id })}>{t.label}</button>
        ))}
      </div>
      <div role="tabpanel" aria-label={tabs.find((t) => t.id === current)?.label}>
        {current === 'system' && (<>
          <p className="muted lead">This machine’s firmware and software versions, and whether fwupd has firmware updates for it.</p>
          <SystemInfo />
        </>)}
        {current === 'gateway' && (<>
          <p className="muted lead">Clients use one OpenAI-compatible address for every running model.</p>
          <div className="bare-head"><GatewayCard /></div>
        </>)}
        {current === 'slots' && (<>
          <p className="muted lead">Running models take turns on the GPU instead of all decoding at once. More tokens per second overall, and every model gets an even share.</p>
          <GpuSlots />
        </>)}
        {current === 'hf' && (<>
          <p className="muted lead">A token lets DGX-kit look up and download gated or private models. Without one, public models still work.</p>
          <HfToken />
        </>)}
        {current === 'images' && (<>
          <p className="muted lead">The Docker images models run in. Each model can also pick its own image in its settings.</p>
          <Images />
        </>)}
        {current === 'folders' && (<>
          <p className="muted lead">Where DGX-kit looks for models on this machine.</p>
          {lib.data ? <Folders key={lib.data.paths.join('\n')} lib={lib.data} onSaved={lib.reload} /> : <Pending error={lib.error} what="the model folders" />}
        </>)}
        {current === 'import' && (<>
          <p className="muted lead">Turns llmctl .conf files into DGX-kit models: same folders, image, draft and options. Nothing starts, and llmctl’s containers are left alone.</p>
          <ImportLlmctl />
        </>)}
        {current === 'disk' && (<>
          <p className="muted lead">The only place model files can be deleted. Removing a model from its menu never touches its files.</p>
          <DiskDelete />
        </>)}
        {current === 'activity' && (<>
          <p className="muted lead">What DGX-kit has done on this machine, newest first.</p>
          <Log canChangePassword={false} />
        </>)}
        {current === 'password' && (<>
          <p className="muted lead">Changing it signs every other browser out.</p>
          <ChangePassword />
        </>)}
      </div>
    </div>
  )
}

type HfConf = { set: boolean; hint: string | null; source: string | null }

function HfToken() {
  const conf = usePoll<HfConf>('/api/settings/hf', 60000)
  const [token, setToken] = useState('')
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null)
  const put = (json: object, done: (r: HfConf & { checked?: boolean; account?: string }) => string) =>
    api<HfConf & { checked?: boolean; account?: string }>('/api/settings/hf', { method: 'PUT', json })
      .then((r) => { setToken(''); setMsg({ text: done(r) }); toast(done(r)); conf.reload() })
      .catch((e: Error) => setMsg({ text: e.message, bad: true }))
  if (!conf.data) return <Pending error={conf.error} what="the Hugging Face token" />
  const c = conf.data
  return (
    <section className="card wide">
      <label>Access token
        <span className="row">
          <input className="grow" type="password" autoComplete="off" value={token} placeholder={c.set ? `saved ${c.hint ?? ''}${c.source === 'environment' ? ' (from the service’s environment)' : ''}` : 'hf_…'}
            onChange={(e) => setToken(e.target.value)} />
          <button className="primary" disabled={!token.trim()} onClick={() => put({ token }, (r) => r.checked ? `Saved. Signed in as ${r.account ?? 'your account'}.` : 'Saved, but Hugging Face could not be reached to check it.')}>Save</button>
          {c.set && c.source === 'settings' && <button className="ghost" onClick={() => put({ clear: true }, () => 'Removed.')}>Remove</button>}
        </span>
        <small className="muted">Kept in DGX-kit’s state folder, readable only by its service. It is checked with Hugging Face when you save.</small>
      </label>
      {msg && <p className={msg.bad ? 'bad' : 'muted'}>{msg.text}</p>}
    </section>
  )
}

type SlotsConf = { enabled: boolean; quantum: number; min_requests: number; end_on_finish: boolean; min_slot: number; active: boolean; engaged: boolean; owner: string | null; backend: string | null; error: string | null }

/** The GPU time slot switch and the slot length. The engines of waiting models are frozen in place (their cache and open
 *  streams survive), so this never restarts anything; it takes effect at once and is kept across dashboard restarts. */
function GpuSlots() {
  const conf = usePoll<SlotsConf>('/api/settings/slots', 2000)
  const [q, setQ] = useState<string | null>(null)
  const [m, setM] = useState<string | null>(null)
  const [ms, setMs] = useState<string | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  if (!conf.data) return <Pending error={conf.error} what="the GPU slot settings" />
  const c = conf.data
  const put = (json: object) => api<SlotsConf>('/api/settings/slots', { method: 'PUT', json })
    .then((r) => { setQ(null); setM(null); setMs(null); setMsg(null); toast(r.enabled ? `GPU slots on, ${r.quantum} s each.` : 'GPU slots off.'); conf.reload() })
    .catch((e: Error) => setMsg(e.message))
  const changed = (v: string | null, cur: number) => v != null && v.trim() !== '' && Number(v) !== cur
  const dirty = changed(q, c.quantum) || changed(m, c.min_requests) || changed(ms, c.min_slot)
  const save = () => put({ ...(changed(q, c.quantum) ? { quantum: Number(q) } : {}), ...(changed(m, c.min_requests) ? { min_requests: Number(m) } : {}), ...(changed(ms, c.min_slot) ? { min_slot: Number(ms) } : {}) })
  return (
    <section className="card wide">
      <label className="check"><input type="checkbox" checked={c.enabled} onChange={(e) => put({ enabled: e.target.checked })} /> Running models take turns on the GPU</label>
      <p className="muted small">On a GB10 every model shares one memory bus, so three models decoding together give fewer tokens per second than one at a time. With slots on, one model runs while the others that have work wait, frozen with their cache intact; a model with nothing to do is never frozen. Short slots keep first tokens quick; longer ones switch less.</p>
      <label>Slot length, seconds
        <span className="row">
          <input className="grow" type="number" min={0.5} max={60} step={0.5} value={q ?? String(c.quantum)} onChange={(e) => setQ(e.target.value)} />
        </span>
        <small className="muted">2 s measured best on a DGX Spark: first tokens in well under a second, about 20% more output than without slots.</small>
      </label>
      <label>Only from this many requests in flight
        <span className="row">
          <input className="grow" type="number" min={1} max={64} step={1} value={m ?? String(c.min_requests)} onChange={(e) => setM(e.target.value)} />
        </span>
        <small className="muted">Slots pay when a model decodes several requests in a turn, and each new request waits up to two slots for its first token. Under this many requests across the busy models (and with fewer than two models busy) the engines run concurrently as before.</small>
      </label>
      <label className="check"><input type="checkbox" checked={c.end_on_finish} onChange={(e) => put({ end_on_finish: e.target.checked })} /> End a slot as soon as the model finishes a request</label>
      <label>…but not before this many seconds
        <span className="row">
          <input className="grow" type="number" min={0.1} max={60} step={0.1} value={ms ?? String(c.min_slot)} onChange={(e) => setMs(e.target.value)} disabled={!c.end_on_finish} />
          <button className="primary" disabled={!dirty} onClick={save}>Save</button>
        </span>
        <small className="muted">Off: every slot lasts the slot length. On: a slot ends at the first finished request after this minimum, and never later than the slot length, so slots stretch for long answers and shorten for short ones. The agent whose request just finished sends its next one while its model still has the GPU. It changes who waits, not how much the box produces.</small>
      </label>
      {c.enabled && <p className="muted small">{!c.active ? 'On, but not running (read-only mode?)' : !c.engaged ? 'On; running concurrently, under the threshold' : c.owner ? `${c.owner} has the GPU now` : 'On; no model is busy'}{c.active && c.backend ? ` · freezing via ${c.backend}` : ''}</p>}
      {c.error && <p className="bad">▲ {c.error}</p>}
      {msg && <p className="bad">{msg}</p>}
    </section>
  )
}

/** What a section shows before its data arrives, or when it can't be read. */
export function Pending({ error, what }: { error: string | null; what: string }) {
  return <p className={error ? 'bad' : 'muted'}>{error ? `Couldn’t read ${what}: ${error}` : `Reading ${what}…`}</p>
}

type ConfItem = { file: string; error?: string; exists?: boolean; notes?: string[]; port?: string
  recipe?: { name: string; image: string | null; path: string | null; repo: string; draft_path: string | null; max_context: number | null; notes: string } }

function ImportLlmctl() {
  const [folder, setFolder] = useState('~/llm-ctl/models')
  const [items, setItems] = useState<ConfItem[] | null>(null)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null)
  const read = () => api<{ items: ConfItem[] }>(`/api/import/llmctl?folder=${encodeURIComponent(folder)}`)
    .then((r) => { setItems(r.items); setPicked(new Set(r.items.filter((i) => i.recipe).map((i) => i.file))); setMsg(null) })
    .catch((e: Error) => setMsg({ text: e.message, bad: true }))
  const run = () => api<{ imported: string[]; skipped: { file: string; why: string }[] }>('/api/import/llmctl', { method: 'POST', json: { files: [...picked] } })
    .then((r) => { setMsg({ text: `Imported ${r.imported.join(', ') || 'nothing'}.${r.skipped.length ? ` Skipped: ${r.skipped.map((x) => `${x.file.split('/').pop()} (${x.why})`).join(', ')}.` : ''}` }); read() })
    .catch((e: Error) => setMsg({ text: e.message, bad: true }))
  // Read the default folder straight away, so importing everything is one click.
  useEffect(() => { read() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  const importable = (items ?? []).filter((i) => i.recipe).map((i) => i.file)
  const all = importable.length > 0 && importable.every((f) => picked.has(f))
  const flip = (f: string) => setPicked((p) => { const n = new Set(p); if (n.has(f)) n.delete(f); else n.add(f); return n })
  const short = (p: string | null) => p?.replace(/^\/home\/[^/]+/, '~') ?? ''
  return (
    <section className="card wide">
      <div className="row">
        <label className="grow">Folder with .conf files<input value={folder} onChange={(e) => setFolder(e.target.value)} /></label>
        <button onClick={read}>Read folder</button>
      </div>
      {items && (items.length === 0 ? <p className="muted">No .conf files there.</p> : (
        <table className="import-table">
          <colgroup><col style={{ width: 32 }} /><col /><col style={{ width: 320 }} /></colgroup>
          <tbody>
            {items.map((i) => (
              <tr key={i.file}>
                <td><input type="checkbox" aria-label={`Import ${i.file}`} disabled={!i.recipe} checked={picked.has(i.file)} onChange={() => flip(i.file)} /></td>
                <td>
                  <strong>{i.recipe?.name ?? i.file.split('/').pop()}</strong>
                  {i.exists && <span className="muted"> · already imported; importing again replaces it (the old one stays in its history)</span>}
                  {i.error && <div className="bad">{i.error}</div>}
                  {i.recipe?.notes && <div className="muted small">{i.recipe.notes}</div>}
                  {!!i.notes?.length && <div className="muted small">{i.notes.join('; ')}</div>}
                </td>
                <td className="muted small">
                  {i.recipe && <>
                    <div>{short(i.recipe.path) || i.recipe.repo}</div>
                    {i.recipe.draft_path && <div>draft {short(i.recipe.draft_path)}</div>}
                    <div>{i.recipe.image}{i.recipe.max_context ? ` · ctx ${i.recipe.max_context.toLocaleString()}` : ''}</div>
                  </>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
      {items && items.length > 0 && (
        <div className="row">
          <label className="check"><input type="checkbox" checked={all} disabled={!importable.length} onChange={() => setPicked(new Set(all ? [] : importable))} /> Select all ({importable.length})</label>
          <button className="primary" disabled={!picked.size} onClick={run}>{all ? `Import all ${picked.size}` : `Import ${picked.size} selected`}</button>
        </div>
      )}
      {msg && <p className={msg.bad ? 'bad' : 'note'}>{msg.text}</p>}
    </section>
  )
}
