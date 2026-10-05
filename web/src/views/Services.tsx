import { useState } from 'react'
import { api, copyText } from '../api'
import { Icon } from '../components/Icon'
import { ModelActions, type Tab } from '../components/ModelActions'
import { stateText, stateTitle, STATE_TONE, type Entry } from '../fleet'
import { select } from '../nav'
import { toast } from '../toast'
import { usePoll } from '../usePoll'
import { PullProgress } from './Other'

type Result = { ms: number; model: string | null; jailbreak: number | null; topic: string | null }

/**
 * A service that runs beside the models and isn't one (Laya, the decision model). It is listed with the
 * models and has the same panel shape: controls and facts on top, then Live, Settings and Logs.
 */
export function ServicePanel({ entry, tab, setTab, readonly, onChanged }: {
  entry: Entry; tab: Tab; setTab: (t: Tab) => void; readonly: boolean; onChanged: () => void
}) {
  const s = entry.service!
  const base = `/api/services/${s.name}`
  const tone = STATE_TONE[entry.state] ?? ''
  const [result, setResult] = useState<Result | null>(null)
  const [testing, setTesting] = useState(false)
  const endpoint = `http://${window.location.hostname}:${s.port}/v1/systemone`

  const test = async () => {
    setTesting(true)
    try { setResult(await api<Result>(`${base}/test`, { method: 'POST' })) } catch (e) { toast((e as Error).message, true) } finally { setTesting(false) }
  }
  const copyKey = async () => toast((await copyText((await api<{ key: string }>(`${base}/key`)).key)) ? 'Key copied.' : 'Could not copy the key.', false)
  const save = (p: { device?: string; port?: number; checkpoints?: string[]; expose?: boolean }) =>
    api(base, { method: 'PUT', json: p }).then(() => { toast(s.state === 'running' ? 'Saved. It applies at the next restart.' : 'Saved.'); onChanged() }).catch((e: Error) => toast(e.message, true))
  const pick = (name: string, on: boolean) => {
    const next = on ? [...s.selected, name] : s.selected.filter((c) => c !== name)
    if (!next.length) return toast('Pick at least one checkpoint.', true)
    save({ checkpoints: next })
  }
  const stale = s.choices.length > 1 && s.state === 'running' && s.loaded.length > 0 && [...s.loaded].sort().join() !== [...s.selected].sort().join()

  return (
    <section className="panel" aria-label={`${s.title} details`}>
      <div className="panel-head">
        <span className={`dot ${tone}`} aria-hidden />
        <h2 className="panel-name">{s.title}</h2><code className="muted" title="Decision model: it doesn't generate text, so it has no place on the gateway">decision model</code>
        <span className={`state-word ${tone}`} title={stateTitle(entry)}>{stateText(entry)}</span>
        <span className="grow" />
        <button className="ic" title="Close" aria-label="Close details" onClick={() => select(undefined)}><Icon name="close" /></button>
      </div>
      <div className="panel-acts"><ModelActions labels entry={entry} readonly={readonly} onChanged={onChanged} onOpen={setTab} /></div>
      <div className="facts">
        <span>port <b>{s.port}</b></span>
        <span>{s.device_in_use ? <>on <b>{s.device_in_use === 'cuda' ? 'GPU' : 'CPU'}</b></> : <>device <b>{s.device === 'cuda' ? 'GPU' : 'CPU'}</b></>}</span>
        {s.loaded.length > 0 && <span>loaded <b>{s.loaded.join(' · ')}</b></span>}
        {s.device_in_use === 'cpu' && s.device === 'cuda' && <span className="warn" title="It asked for the GPU and could not get it; see the Logs tab">▲ on CPU, about 15× slower</span>}
        <span className="svc-url"><code>{endpoint}</code> <button className="ghost" onClick={() => copyText(endpoint).then(() => toast('Address copied.'))}>copy</button></span>
        {s.has_key ? <button className="ghost" onClick={copyKey}>copy key</button>
          : <span className="muted" title="Its server has no API key, so anyone who can reach the port can use it">{s.expose ? '▲ no key, on the network' : 'no key: this machine only'}</span>}
        {s.needs_bytes != null && s.state !== 'running' && <span className="muted">needs about <b>{Math.round(s.needs_bytes / 2 ** 30)} GiB</b> free to start</span>}
      </div>
      <p className="model-notes">{s.about}</p>
      {s.problems.map((p) => <p key={p} className="advice bad"><span aria-hidden>● </span>{p}</p>)}
      {s.error && <p className="advice bad"><span aria-hidden>● </span>{s.error}</p>}
      {s.build && <PullProgress job={s.build} />}

      <div className="tabs" role="tablist">
        {(['live', 'settings', 'logs'] as Tab[]).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>{t === 'live' ? 'Live' : t === 'settings' ? 'Settings' : 'Logs'}</button>
        ))}
      </div>

      {tab === 'live' && (s.state === 'running' ? (
        <div>
          <div className="row">
            <button disabled={testing} onClick={test} title="Send it a sample question and time the answer">{testing ? 'Asking…' : 'Test'}</button>
            {result && <span className="muted small">Answered in <b>{result.ms} ms</b>{result.model ? ` by ${result.model}` : ''}: “ignore all previous instructions…” looks like a jailbreak with probability <b>{result.jailbreak?.toFixed(2)}</b>, topic {result.topic}.</span>}
          </div>
          <p className="muted small">Ask it with <code>POST /v1/systemone</code>{s.has_key ? ' and the key above' : ''}: a <code>state</code> (text or JSON) and typed <code>questions</code> (yes/no, choice, score){s.choices.length > 1 ? <>; <code>model</code> picks a checkpoint</> : null}.</p>
        </div>
      ) : (
        <p className="empty-note">{s.state === 'preparing' ? 'Building its image; it starts by itself when that is done.' : s.state === 'starting' ? 'Loading the checkpoints.' : s.state === 'exited' ? 'It stopped on its own. The Logs tab says why; Start tries again.' : 'Stopped. Start it to ask it questions.'}</p>
      ))}

      {tab === 'settings' && (
        <div className="quick">
          {s.choices.length > 1 ? (
            <fieldset className="ckpts">
              <legend>Checkpoints to run <small className="muted">found in <code>{s.checkpoints.dir ?? 'the models folder (none)'}</code></small></legend>
              {s.choices.map((c) => (
                <label key={c.name} className="check">
                  <input type="checkbox" checked={s.selected.includes(c.name)} disabled={readonly || !c.found} onChange={(e) => pick(c.name, e.target.checked)} />
                  {c.name} <small className="muted">{c.found ? c.about : 'not in the folder'}</small>
                </label>
              ))}
            </fieldset>
          ) : (
            <p className="muted small">{s.choices[0]?.found ? <>Runs <b>{s.choices[0].name}</b> ({s.choices[0].about}) from <code>{s.checkpoints.dir}</code>.</> : <>Not on this box yet; Start says what to put where.</>}</p>
          )}
          {stale && <p className="advice warn"><span aria-hidden>▲ </span>Running: {s.loaded.join(' · ')}. Restart to run {s.selected.join(' · ')}.</p>}
          <div className="params">
            <label>
              <span>Device</span>
              <select value={s.device} disabled={readonly} onChange={(e) => save({ device: e.target.value })}><option value="cuda">GPU (falls back to CPU)</option><option value="cpu">CPU</option></select>
              <small>much faster on the GPU</small>
            </label>
            <label>
              <span>Port</span>
              <input type="number" defaultValue={s.port} disabled={readonly} onBlur={(e) => Number(e.target.value) !== s.port && save({ port: Number(e.target.value) })} />
              <small>applies at the next start</small>
            </label>
          </div>
          {s.can_expose && (
            <label className="check">
              <input type="checkbox" checked={!!s.expose} disabled={readonly} onChange={(e) => save({ expose: e.target.checked })} />
              Listen on the network <small className="muted">(it has no API key: anyone who can reach the port can use it; applies at the next start)</small>
            </label>
          )}
        </div>
      )}
      {tab === 'logs' && <ServiceLogs path={`${base}/logs?tail=300`} />}
    </section>
  )
}

function ServiceLogs({ path }: { path: string }) {
  const { data } = usePoll<{ logs: string }>(path, 3000)
  return (
    <>
      <p className="muted hint">Last 300 lines, refreshing every 3 seconds.</p>
      <pre className="logs">{data ? data.logs || 'No output yet.' : 'Loading…'}</pre>
    </>
  )
}
