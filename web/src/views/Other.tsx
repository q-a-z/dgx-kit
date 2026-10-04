import { Pending } from './Settings'
import { useState } from 'react'
import { api, fmtTime, type Action, type Image } from '../api'
import { usePoll } from '../usePoll'
import { ChangePassword } from '../Login'
import { toast } from '../toast'

const LABELS: Record<string, string> = { vllm: 'vLLM', sglang: 'SGLang', llamacpp: 'llama.cpp', litellm: 'LiteLLM gateway' }

/** A bar and the words for a pull or build that is running (or the reason it failed). */
export function PullProgress({ job }: { job: { kind: string; state: string; error: string | null; tail: string[]; progress?: number | null } }) {
  if (job.state === 'done' && !job.error) return null
  return (
    <div className="pull-progress">
      {job.state === 'running' && <progress className="pull" max={1} value={job.progress ?? undefined} aria-label={`${job.kind} progress`} />}
      <pre className="logs">{[...job.tail, job.error ?? ''].filter(Boolean).join('\n') || `${job.kind} started`}</pre>
    </div>
  )
}

function ImageRow({ i, onChange }: { i: Image; onChange: () => void }) {
  const [edit, setEdit] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const running = i.job?.state === 'running'
  const others = (i.local ?? []).filter((t) => t !== i.image && !i.builds.some((b) => b.image === t))
  const call = (path: string, init?: Parameters<typeof api>[1], saved?: string) =>
    api(path, init).then(() => { if (saved) toast(saved); onChange() }).catch((e: Error) => { setErr(e.message); toast(e.message, true) })
  // Pointing an engine at a tag the box doesn't have starts the pull at once, so its progress can be watched here.
  const choose = (image: string | null) =>
    api<{ image: string }>(`/api/images/${i.engine}`, { method: 'PUT', json: { image } })
      .then((r) => api<{ ready: boolean }>(`/api/images/status?image=${encodeURIComponent(r.image)}`).then((st) => {
        toast(`${LABELS[i.engine] ?? i.engine} will use ${r.image}.`)
        return st.ready ? undefined : api(`/api/images/${i.engine}/pull`, { method: 'POST' }).then(() => toast(`Pulling ${r.image}…`))
      }))
      .then(onChange).catch((e: Error) => { setErr(e.message); toast(e.message, true) })

  return (
    <section className="card wide">
      <div className="row">
        <h2 className="grow">{LABELS[i.engine] ?? i.engine}</h2>
        <span className={`pill ${i.ready ? 'ok' : others.length ? '' : 'warn'}`}>{running ? `${i.job!.kind}ing…` : i.ready ? (i.source === 'found' ? 'found on this box' : 'on this box') : 'not pulled'}</span>
      </div>
      <p>
        <code>{i.image}</code>
        {i.source === 'found' && <span className="muted"> · already here, so DGX-kit uses it instead of pulling {i.default}</span>}
        {i.source === 'chosen' && <span className="muted"> (default {i.default})</span>}
      </p>
      {others.length > 0 && (
        <div className="found-local">
          <p className="muted">Other {LABELS[i.engine] ?? i.engine} images on this box:</p>
          {others.map((t) => (
            <div key={t} className="row"><code className="grow">{t}</code>
              <button onClick={() => choose(t)}>Use this</button></div>
          ))}
        </div>
      )}
      {edit !== null && (
        <div className="row">
          <input className="grow" value={edit} onChange={(e) => setEdit(e.target.value)} placeholder="registry/name:tag" />
          <button className="primary" onClick={() => { choose(edit.trim() || null); setEdit(null) }}>Use this tag</button>
          <button onClick={() => setEdit(null)}>Cancel</button>
        </div>
      )}
      <div className="row">
        {i.source !== 'found' && <button disabled={running} onClick={() => call(`/api/images/${i.engine}/pull`, { method: 'POST' })}>{i.ready ? 'Pull again' : 'Pull'}</button>}
        {edit === null && <button onClick={() => setEdit(i.image)}>Change tag</button>}
        {i.source === 'chosen' && <button onClick={() => choose(null)}>Back to default</button>}
      </div>
      {i.job && <PullProgress job={i.job} />}
      {i.builds.length > 0 && (
        <table className="builds">
          <tbody>
            {i.builds.map((b) => {
              const busy = b.job?.state === 'running'
              // llama.cpp has one build and it replaces the engine image; vLLM builds are picked per model.
              const q = i.builds.length === 1 ? '?make_default=true' : ''
              return (
                <tr key={b.id}>
                  <td>
                    <div>{b.about}</div>
                    <div className="muted">
                      <code>{b.image}</code> · {i.engine !== 'vllm' ? 'compiled on this box' : b.patches.length ? <span title={b.patches.join('\n')}>{b.patches.length} patch{b.patches.length > 1 ? 'es' : ''}: {b.patches.map((x) => x.split('/').pop()!.replace(/^\d+-|\.patch$/g, '')).join(', ')}</span> : 'no patches added yet'}
                    </div>
                    {b.job && <PullProgress job={b.job} />}
                  </td>
                  <td className="num">{busy ? 'building…' : b.ready ? 'built' : 'not built'}</td>
                  <td className="num">
                    <button disabled={busy} onClick={() => call(`/api/images/builds/${b.id}${q}`, { method: 'POST' })}>{b.ready ? 'Rebuild' : 'Build'}</button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
      {err && <p className="bad">{err}</p>}
    </section>
  )
}

export function Images() {
  const [busy, setBusy] = useState(false)
  const { data, error, reload } = usePoll<Image[]>('/api/images', busy ? 2000 : 15000)
  const anyJob = !!data?.some((i) => i.job?.state === 'running' || i.builds.some((b) => b.job?.state === 'running'))
  if (anyJob !== busy) setBusy(anyJob)
  const [note, setNote] = useState<string | null>(null)
  const clean = async () => {
    const r = await api<{ removed: string[] }>('/api/images/clean', { method: 'POST' }).catch((e: Error) => ({ removed: [], error: e.message }))
    setNote('error' in r ? String(r.error) : r.removed.length ? `Removed ${r.removed.join(', ')}.` : 'Nothing to remove.')
    reload()
  }
  return (
    <div className="grid">
      <p className="muted wide">
        Tag changes apply the next time a model starts. <button onClick={clean}>Remove unused images</button> {note}
      </p>
      {data ? data.map((i) => <ImageRow key={i.engine} i={i} onChange={reload} />) : <Pending error={error} what="the engine images" />}
    </div>
  )
}

export function Log({ canChangePassword }: { canChangePassword: boolean }) {
  const { data } = usePoll<Action[]>('/api/log', 3000)
  return (
    <div className="grid">
    {canChangePassword && <ChangePassword />}
    <section className="card wide">
      <h2>What DGX-kit did</h2>
      {!data?.length ? <p className="muted">Nothing yet.</p> : (
        <table>
          <tbody>
            {[...data].reverse().map((a, i) => (
              <tr key={i}><td className="muted">{fmtTime(a.t, true)}</td><td>{a.action}</td><td>{a.detail}</td></tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
    </div>
  )
}
