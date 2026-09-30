import { Pending } from './Settings'
import { useState } from 'react'
import { api, type Action, type Image } from '../api'
import { usePoll } from '../usePoll'
import { ChangePassword } from '../Login'

const LABELS: Record<string, string> = { vllm: 'vLLM', sglang: 'SGLang', llamacpp: 'llama.cpp', litellm: 'LiteLLM gateway' }

function ImageRow({ i, onChange }: { i: Image; onChange: () => void }) {
  const [edit, setEdit] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const running = i.job?.state === 'running'
  const others = (i.local ?? []).filter((t) => t !== i.image && !i.builds.some((b) => b.image === t))
  const call = (path: string, init?: Parameters<typeof api>[1]) =>
    api(path, init).then(onChange).catch((e: Error) => setErr(e.message))

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
              <button onClick={() => call(`/api/images/${i.engine}`, { method: 'PUT', json: { image: t } })}>Use this</button></div>
          ))}
        </div>
      )}
      {edit !== null && (
        <div className="row">
          <input className="grow" value={edit} onChange={(e) => setEdit(e.target.value)} placeholder="registry/name:tag" />
          <button className="primary" onClick={() => { call(`/api/images/${i.engine}`, { method: 'PUT', json: { image: edit.trim() || null } }); setEdit(null) }}>Use this tag</button>
          <button onClick={() => setEdit(null)}>Cancel</button>
        </div>
      )}
      <div className="row">
        {i.source !== 'found' && <button disabled={running} onClick={() => call(`/api/images/${i.engine}/pull`, { method: 'POST' })}>{i.ready ? 'Pull again' : 'Pull'}</button>}
        {edit === null && <button onClick={() => setEdit(i.image)}>Change tag</button>}
        {i.source === 'chosen' && <button onClick={() => call(`/api/images/${i.engine}`, { method: 'PUT', json: { image: null } })}>Back to default</button>}
      </div>
      {i.job && (i.job.state !== 'done' || i.job.error) && (
        <pre className="logs">{[...i.job.tail, i.job.error ?? ''].filter(Boolean).join('\n') || `${i.job.kind} started`}</pre>
      )}
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
                    {b.job && (b.job.state !== 'done' || b.job.error) && (
                      <pre className="logs">{[...b.job.tail, b.job.error ?? ''].filter(Boolean).join('\n') || 'build started'}</pre>
                    )}
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
              <tr key={i}><td className="muted">{new Date(a.t * 1000).toLocaleTimeString()}</td><td>{a.action}</td><td>{a.detail}</td></tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
    </div>
  )
}
