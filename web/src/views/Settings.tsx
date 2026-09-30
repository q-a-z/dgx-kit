import { useEffect, useState } from 'react'
import { api } from '../api'
import { ChangePassword } from '../Login'
import { usePoll } from '../usePoll'
import { Folders } from './Library'
import { DiskDelete } from './DiskDelete'
import { GatewayCard } from './Models'
import { Images, Log } from './Other'

type Lib = Parameters<typeof Folders>[0]['lib']

/** Set-up-once things: how clients reach the models, which engine images run them, where models live. */
export function Settings({ canChangePassword }: { canChangePassword: boolean }) {
  const lib = usePoll<Pick<Lib, 'paths' | 'missing'>>('/api/library/paths', 30000)
  return (
    <div className="page narrow-page settings">
      
      <h1>Settings</h1>
      <nav className="toc" aria-label="On this page">
        <a href="#gateway" onClick={jump}>Gateway</a><a href="#hf" onClick={jump}>Hugging Face</a><a href="#images" onClick={jump}>Engine images</a>
        <a href="#folders" onClick={jump}>Model folders</a><a href="#disk" onClick={jump}>Delete from disk</a><a href="#import" onClick={jump}>Import</a><a href="#activity" onClick={jump}>Activity</a>
        {canChangePassword && <a href="#password" onClick={jump}>Password</a>}
      </nav>
      <h2 id="gateway">Gateway</h2>
      <p className="muted lead">Clients use one OpenAI-compatible address for every running model.</p>
      <div className="bare-head"><GatewayCard /></div>
      <h2 id="hf">Hugging Face</h2>
      <p className="muted lead">A token lets DGX-kit look up and download gated or private models. Without one, public models still work.</p>
      <HfToken />
      <h2 id="images">Engine images</h2>
      <p className="muted lead">The Docker images models run in. Each model can also pick its own image in its settings.</p>
      <Images />
      <h2 id="folders">Model folders</h2>
      {lib.data ? <Folders key={lib.data.paths.join('\n')} lib={lib.data} onSaved={lib.reload} /> : <Pending error={lib.error} what="the model folders" />}
      <h2 id="disk">Delete from disk</h2>
      <p className="muted lead">The only place model files can be deleted. Removing a model from its menu never touches its files.</p>
      <DiskDelete />
      <h2 id="import">Import from llmctl</h2>
      <p className="muted lead">Turns llmctl .conf files into DGX-kit models: same folders, image, draft and options. Nothing starts, and llmctl’s containers are left alone.</p>
      <ImportLlmctl />
      <h2 id="activity">Activity</h2>
      <Log canChangePassword={false} />
      {canChangePassword && <><h2 id="password">Password</h2><ChangePassword /></>}
    </div>
  )
}

// In-page links without touching the hash router.
function jump(e: React.MouseEvent<HTMLAnchorElement>) {
  e.preventDefault()
  document.getElementById(e.currentTarget.getAttribute('href')!.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

type HfConf = { set: boolean; hint: string | null; source: string | null }

function HfToken() {
  const conf = usePoll<HfConf>('/api/settings/hf', 60000)
  const [token, setToken] = useState('')
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null)
  const put = (json: object, done: (r: HfConf & { checked?: boolean; account?: string }) => string) =>
    api<HfConf & { checked?: boolean; account?: string }>('/api/settings/hf', { method: 'PUT', json })
      .then((r) => { setToken(''); setMsg({ text: done(r) }); conf.reload() })
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
