import { usePoll } from '../usePoll'
import { useEffect, useRef, useState } from 'react'
import { useModelActions } from '../actions'
import { toast } from '../toast'
import { PullProgress } from './Other'
import { api, copyText, fmtBytes, fmtNum, type ModelRow, type Plan, type Recipe, type Snapshot } from '../api'
import type { Mark } from '../components/Chart'
import { Icon } from '../components/Icon'
import { ModelActions, type Tab } from '../components/ModelActions'
import { base, ENGINES, prettyName, STATE_LABEL, STATE_TONE, type Entry } from '../fleet'
import type { Alert } from '../health'
import { select } from '../nav'
import type { Layout } from '../tiles/layout'
import type { Slo } from '../tiles/modelTiles'
import { PANEL_TILES } from '../tiles/panelTiles'
import { TileGrid } from '../tiles/TileGrid'
import { Bench } from './Bench'
import { DownloadProgress, Editor } from './Models'

export type Preview = { name: string; bytes: number } | null

/**
 * One model, beside the thing you picked it from: its controls next to its name, then Live, Settings or Logs.
 * The panel has a fixed width; it never takes the whole page.
 */
export function ModelPanel({ entry, tab, setTab, latest, history, readonly, layout, update, slo, setSlo, marks, alert, onChanged, onPreview }: {
  entry: Entry; tab: Tab; setTab: (t: Tab) => void; latest: Snapshot; history: Snapshot[]; readonly: boolean
  layout: Layout; update: (p: Partial<Layout>) => void; slo: Slo | null; setSlo: (s: Slo) => void; marks: Mark[]
  alert?: Alert; onChanged: () => void; onPreview: (p: Preview) => void
}) {
  const [arrange, setArrange] = useState(false)
  const r = entry.row
  const live = entry.live
  const engine = r?.engine ?? entry.ext?.engine ?? ''
  const port = r?.container?.port ?? entry.ext?.port
  const ctx = (live?.context_tokens as number | undefined) ?? r?.max_context
  const draft = r?.draft_repo ?? entry.ext?.draft
  const qc = entry.running ? r?.quick : null  // the ten-second check taken when it started
  const pid = latest.gpu?.processes.find((p) => p.key === entry.name || p.model === entry.name)?.pid
  const tone = STATE_TONE[entry.state] ?? ''

  return (
    <section className="panel" aria-label={`${entry.name} details`}>
      <div className="panel-head">
        <span className={`dot ${tone}`} aria-hidden />
        <h2 className="panel-name">{prettyName(entry)}</h2>{prettyName(entry) !== entry.name && <code className="muted" title="Name clients use on the gateway">{entry.name}</code>}
        <span className={`state-word ${tone}`}>{STATE_LABEL[entry.state] ?? entry.state}</span>
        <span className="grow" />
        <button className="ic" title="Close" aria-label="Close details" onClick={() => select(undefined)}><Icon name="close" /></button>
      </div>
      <div className="panel-acts">
        <ModelActions labels entry={entry} readonly={readonly} onChanged={onChanged} onOpen={setTab} />
        {entry.managed && <More entry={entry} readonly={readonly} onChanged={onChanged} />}
      </div>
      <div className="facts">
        {(r?.repo ?? entry.ext?.model) && <span title={r?.repo ?? entry.ext?.model ?? ''}>{base(r?.repo ?? entry.ext?.model)}</span>}
        {engine && <span>{ENGINES[engine] ?? engine}{r?.quantization ? ` · ${r.quantization}` : ''}</span>}
        {entry.memBytes != null && <span><b>{fmtBytes(entry.memBytes)}</b></span>}
        {ctx ? <span>ctx <b>{fmtNum(ctx)}</b></span> : null}
        {draft && <span>draft <b>{base(draft)}</b></span>}
        {port ? <span>port <b>{port}</b></span> : null}
        {entry.managed && r && <span title="Whether it is listed on the gateway while it runs">gateway <b>{r.publish ? 'published' : 'not published'}</b></span>}
        {pid ? <span>PID <b>{pid}</b></span> : null}
        {qc && (qc.decode_mean_tps != null || qc.prefill_mean_tps != null
          ? <span title="Ten-second check taken when it started">at start: decode <b>{fmtNum(qc.decode_mean_tps ?? 0, 0)} t/s</b> · prefill <b>{fmtNum(qc.prefill_mean_tps ?? 0, 0)} t/s</b></span>
          : <span className="muted">start check {qc.skipped ? `skipped: ${qc.skipped}` : `failed: ${qc.error ?? 'no numbers'}`}</span>)}
        {!entry.managed && <span>started outside DGX-kit</span>}
      </div>
      {r?.notes && <p className="model-notes">{r.notes}</p>}
      {alert && (
        <p className={`advice ${alert.level}`}>
          <span aria-hidden>{alert.level === 'bad' ? '● ' : '▲ '}</span>{alert.text}{alert.advice && <> <span className="muted">{alert.advice}</span></>}
          {alert.advice && entry.managed && tab !== 'settings' && alert.level === 'warn' && <> <button className="link" onClick={() => setTab('settings')}>Adjust settings</button></>}
          {alert.level === 'bad' && tab !== 'logs' && <> <button className="link" onClick={() => setTab('logs')}>Open logs</button></>}
        </p>
      )}
      {entry.download && <DownloadProgress d={entry.download} onChange={onChanged} />}

      <div className="tabs" role="tablist">
        {(['live', 'settings', 'logs', 'bench'] as Tab[]).filter((t) => t === 'bench' ? entry.managed : t !== 'settings' || entry.managed || entry.running).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>
            {t === 'live' ? 'Live' : t === 'settings' ? (entry.managed ? 'Settings' : 'Launch settings') : t === 'bench' ? 'Benchmark' : 'Logs'}
          </button>
        ))}
        <span className="grow" />
        {tab === 'live' && entry.running && <button className="ghost" aria-pressed={arrange} onClick={() => setArrange(!arrange)}>{arrange ? 'Done' : 'Arrange or add charts'}</button>}
      </div>

      {tab === 'live' && (entry.running ? (
        <>
          {arrange && <p className="muted hint">Drag a chart by its title, resize from its corner, ✕ hides it. More charts are listed at the end.</p>}
          <TileGrid defs={PANEL_TILES} tiles={layout.panel} editing={arrange} onChange={(panel) => update({ panel })}
            ctx={{ name: entry.name, live, history, stat: layout.stat ?? 'p95', slo, setSlo, readonly, marks }} />
          {marks.length > 0 && <p className="muted marks-key">Dashed lines mark starts, stops and settings changes.</p>}
        </>
      ) : (
        <p className="empty-note">
          {entry.state === 'nofiles' ? `The weights aren’t at ${entry.row?.path}. Put them there, or add the folder that has them to Model folders in Settings and import it again.`
            : entry.state === 'missing' ? 'The weights aren’t on this box yet. Download them, then start it.'
            : entry.state === 'downloading' || entry.state === 'paused' ? 'Downloading. It can start once the download finishes.'
            : entry.state === 'exited' ? `It crashed${entry.row?.container?.exit_code != null ? ` (exit code ${entry.row.container.exit_code})` : ''}. The Logs tab says why. Stop clears it; Start tries again.`
            : entry.state === 'restarting' ? 'It keeps crashing and Docker keeps restarting it. The Logs tab says why; Stop ends the loop.'
            : 'Stopped. Start it to see live numbers here.'}
        </p>
      ))}
      {tab === 'settings' && r && (
        <QuickSettings key={entry.name} entry={entry} model={r} readonly={readonly} onChanged={onChanged} onPreview={onPreview} />
      )}
      {tab === 'settings' && !entry.managed && <LaunchView name={entry.name} />}
      {tab === 'logs' && <Logs name={entry.name} external={!entry.managed} />}
      {tab === 'bench' && entry.managed && <Bench name={entry.name} ready={entry.state === 'running'} readonly={readonly} />}
    </section>
  )
}

function More({ entry, readonly, onChanged }: { entry: Entry; readonly: boolean; onChanged: () => void }) {
  const a = useModelActions(entry, readonly, onChanged)
  const [fit, setFit] = useState<string | null>(null)
  const menu = useRef<HTMLDetailsElement>(null)
  const close = () => { if (menu.current) menu.current.open = false }
  // A <details> stays open until its own summary is clicked again; a menu should also close when you click
  // anywhere else, press Escape, or pick an item.
  useEffect(() => {
    const away = (e: Event) => { if (menu.current?.open && !menu.current.contains(e.target as Node)) close() }
    const esc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && menu.current?.open) { close(); menu.current.querySelector('summary')?.focus() }
    }
    document.addEventListener('pointerdown', away)
    document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('pointerdown', away); document.removeEventListener('keydown', esc) }
  }, [])
  const check = () => api<Plan>(`/api/models/${encodeURIComponent(entry.name)}/plan`).then((p) => setFit(p.fits
    ? `Fits now: ${fmtNum(p.context_tokens)} context, KV cache ${fmtBytes(p.kv_bytes)}, about ${fmtNum(p.concurrency, 1)} full-length requests at once.`
    : `Doesn’t fit right now: ${p.reason}`))
  return (
    <>
      <details className="more" ref={menu}>
        <summary aria-label="More actions" title="More actions"><Icon name="more" /></summary>
        <div className="menu">
          <button onClick={() => { close(); check() }}>Check it fits now</button>
          <button className="danger" disabled={a.locked || entry.running} title={entry.running ? 'Stop it first' : undefined}
            onClick={async () => { close(); if (await a.remove()) select(undefined) }}>Remove…</button>
        </div>
      </details>
      {fit && <p className="fit-note">{fit} <button className="link" onClick={() => setFit(null)}>OK</button></p>}
    </>
  )
}

const FIELDS: { key: 'max_context' | 'min_concurrency' | 'kv_cache_dtype' | 'num_speculative_tokens'; label: string; hint: string }[] = [
  { key: 'max_context', label: 'Max context', hint: 'tokens per request' },
  { key: 'min_concurrency', label: 'Min requests at once', hint: 'at full context' },
  { key: 'kv_cache_dtype', label: 'KV cache type', hint: 'fp8 holds twice the tokens' },
  { key: 'num_speculative_tokens', label: 'Draft tokens per step', hint: 'speculative decoding' },
]

/** The settings people actually change, with what they'll do shown before anything restarts. */
function QuickSettings({ entry, model, readonly, onChanged, onPreview }: {
  entry: Entry; model: ModelRow; readonly: boolean; onChanged: () => void; onPreview: (p: Preview) => void
}) {
  const [r, setR] = useState<Recipe>(model)
  const [plan, setPlan] = useState<Plan | null>(null)
  const [now, setNow] = useState<Plan | null>(null)
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null)
  const [all, setAll] = useState(false)
  const a = useModelActions(entry, readonly, onChanged)
  const q = encodeURIComponent(model.name)
  const own = entry.running ? entry.memBytes ?? 0 : 0
  const changed = FIELDS.filter((f) => r[f.key] !== model[f.key])
  const timer = useRef<number | undefined>(undefined)
  const body = (x: Recipe) => ({ max_context: x.max_context, min_concurrency: x.min_concurrency, kv_cache_dtype: x.kv_cache_dtype })

  useEffect(() => {
    api<Plan>(`/api/models/${q}/plan?own_bytes=${own}`, { method: 'POST', json: body(model) }).then(setNow).catch(() => {})
  }, [q]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    window.clearTimeout(timer.current)
    if (!changed.length) { setPlan(null); onPreview(null); return }
    timer.current = window.setTimeout(() => {
      api<Plan>(`/api/models/${q}/plan?own_bytes=${own}`, { method: 'POST', json: body(r) }).then((p) => {
        setPlan(p)
        onPreview(p.fits ? { name: model.name, bytes: Math.round((model.weights_bytes + model.draft_weights_bytes) * 1.1) + p.kv_bytes } : null)
      }).catch(() => {})
    }, 250)
  }, [r.max_context, r.min_concurrency, r.kv_cache_dtype, r.num_speculative_tokens]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => onPreview(null), []) // eslint-disable-line react-hooks/exhaustive-deps

  const save = async (then: 'restart' | 'start' | null) => {
    setMsg(null)
    try {
      const { container: _c, live: _l, downloaded: _d, ...rest } = { ...model, ...r } as ModelRow
      await api(`/api/models/${q}`, { method: 'PUT', json: rest })
      onPreview(null)
      if (then === 'restart') await a.restart()
      else if (then === 'start') await a.start()
      setMsg({ text: then ? 'Saved.' : entry.running ? 'Saved. It applies at the next restart.' : 'Saved.' })
      toast(then ? 'Settings saved.' : entry.running ? 'Settings saved. They apply at the next restart.' : 'Settings saved.')
      onChanged()
    } catch (e) {
      setMsg({ text: (e as Error).message, bad: true })
    }
  }
  const set = (k: (typeof FIELDS)[number]['key'], v: string) =>
    setR({ ...r, [k]: k === 'kv_cache_dtype' ? v : v.trim() === '' && k === 'max_context' ? null : Number(v) })
  const show = FIELDS.filter((f) => f.key !== 'num_speculative_tokens' || model.draft_repo)
  const verb = entry.running ? 'restart' : ['stopped', 'exited'].includes(entry.state) ? 'start' : null

  return (
    <div className="quick">
      <div className="params">
        {show.map((f) => {
          const was = model[f.key]
          const is = r[f.key]
          return (
            <label key={f.key} className={is !== was ? 'changed' : ''}>
              <span>{f.label}</span>
              {f.key === 'kv_cache_dtype'
                ? <select value={String(is)} onChange={(e) => set(f.key, e.target.value)}><option>auto</option><option>fp8</option><option>bf16</option></select>
                : <input type="number" step={f.key === 'min_concurrency' ? 0.5 : 1} value={is == null ? '' : String(is)} placeholder="from model" onChange={(e) => set(f.key, e.target.value)} />}
              <small>{is !== was ? `was ${was == null ? 'from model' : typeof was === 'number' ? fmtNum(was) : was}` : f.hint}</small>
            </label>
          )
        })}
      </div>
      {plan && now && (
        <p className={`preview ${plan.fits ? '' : 'bad'}`}>
          {plan.fits ? <>
            After {verb === 'start' ? 'starting' : 'restarting'}: context <b>{fmtNum(now.context_tokens)} → {fmtNum(plan.context_tokens)}</b>,
            room for <b>{fmtNum(now.concurrency, 1)} → {fmtNum(plan.concurrency, 1)}</b> full-length requests at once.
            {' '}KV cache {fmtBytes(plan.kv_bytes)}{plan.reason ? `. ${plan.reason}` : ''}.
          </> : <>Won’t fit: {plan.reason}</>}
        </p>
      )}
      <div className="row">
        {verb && <button className="primary" disabled={!changed.length || a.locked || (plan != null && !plan.fits)} title={a.locked ? a.why ?? undefined : undefined}
          onClick={() => save(verb)}>{verb === 'restart' ? 'Restart' : 'Start'} with {changed.length || 'no'} change{changed.length === 1 ? '' : 's'}</button>}
        <button disabled={!changed.length} onClick={() => save(null)}>Save only</button>
        {changed.length > 0 && <button className="ghost" onClick={() => setR(model)}>Undo</button>}
        <span className="grow" />
        <button className="ghost" aria-expanded={all} onClick={() => setAll(!all)}>{all ? 'Hide all settings' : 'All settings and flags'}</button>
      </div>
      {msg && <p className={msg.bad ? 'bad' : 'muted'}>{msg.text}</p>}
      {model.preparing && (
        <div>
          <p className="muted">Fetching <code>{model.preparing.image}</code>; the model starts by itself when it is here.</p>
          <PullProgress job={{ kind: model.preparing.kind ?? 'pull', state: model.preparing.state ?? 'running', error: model.preparing.error ?? null, tail: model.preparing.tail ?? [], progress: model.preparing.progress }} />
        </div>
      )}
      {all && <Editor model={model} onDone={(t) => { setMsg({ text: t }); toast(t); onChanged() }} />}
    </div>
  )
}

/** The model's log tail, refreshed while open, following the end unless scrolled up. */
export function Logs({ name, external }: { name: string; external: boolean }) {
  const [text, setText] = useState('')
  const pre = useRef<HTMLPreElement>(null)
  const follow = useRef(true)
  useEffect(() => {
    const path = external ? `/api/running/${encodeURIComponent(name)}/logs?tail=300` : `/api/models/${encodeURIComponent(name)}/logs?tail=300`
    let stop = false
    const load = () => api<{ logs: string }>(path).then((r) => !stop && setText(r.logs || 'No output yet.')).catch((e: Error) => !stop && setText(e.message))
    load()
    const id = setInterval(load, 3000)
    return () => { stop = true; clearInterval(id) }
  }, [name, external])
  useEffect(() => {
    if (pre.current && follow.current) pre.current.scrollTop = pre.current.scrollHeight
  }, [text])
  return (
    <>
      <p className="muted hint">Last 300 lines, refreshing every 3 seconds.</p>
      <pre ref={pre} className="logs" onScroll={(e) => {
        const el = e.currentTarget
        follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 30
      }}>{text || 'Loading…'}</pre>
    </>
  )
}

type Launch = { image: string; command: string; options: [string | null, string | null][]; env: [string, string][]; mounts: [string, string][]; ports: string[]; restart: string; shm: number | null }

/** How a model started outside DGX-kit was launched. Read-only: changing it means restarting the container. */
function LaunchView({ name }: { name: string }) {
  const { data, error } = usePoll<Launch>(`/api/running/${encodeURIComponent(name)}/launch`, 60000)
  const [copied, setCopied] = useState(false)
  const [imported, setImported] = useState<string | null>(null)
  const [importErr, setImportErr] = useState<string | null>(null)
  const doImport = () => api<{ name: string; notes: string[] }>(`/api/import/running/${encodeURIComponent(name)}`, { method: 'POST' })
    .then((r) => { setImported(r.name); setImportErr(r.notes.length ? r.notes.join('; ') : null) })
    .catch((e: Error) => setImportErr(e.message))
  if (!data) return <p className={error ? 'bad' : 'muted'}>{error ? `Couldn’t read how it was started: ${error}` : 'Reading how it was started…'}</p>
  const copy = () => copyText(data.command).then((ok) => { if (ok) { setCopied(true); setTimeout(() => setCopied(false), 1500) } })
  return (
    <div className="launch">
      <p className="muted">Started outside DGX-kit, so this is read-only. DGX-kit never restarts it.</p>
      <dl className="launch-facts">
        <dt>Image</dt><dd><code>{data.image}</code></dd>
        <dt>Ports</dt><dd>{data.ports.join(', ') || 'none'}</dd>
        <dt>Restart</dt><dd>{data.restart}</dd>
        {data.mounts.length > 0 && <><dt>Folders</dt><dd>{data.mounts.map(([a, b]) => <div key={b}><code>{a}</code> → <code>{b}</code></div>)}</dd></>}
      </dl>
      <table className="launch-opts">
        <thead><tr><th>Option</th><th>Value</th></tr></thead>
        <tbody>
          {data.options.map(([k, v], i) => <tr key={i}><td><code>{k ?? ''}</code></td><td><code>{v ?? (k ? 'on' : '')}</code></td></tr>)}
          {data.env.map(([k, v]) => <tr key={k}><td><code>{k}</code> <span className="muted">env</span></td><td><code>{v}</code></td></tr>)}
        </tbody>
      </table>
      <div className="row">
        <button className="primary" disabled={!!imported} onClick={doImport}>{imported ? `Imported as ${imported}` : 'Import into DGX-kit'}</button>
        <button onClick={copy}>{copied ? 'Copied' : 'Copy command'}</button>
        {importErr && <span className="bad">{importErr}</span>}
      </div>
      <p className="muted small">Import saves this command as a DGX-kit model with the same folders, image and options. This container keeps running; stop it here or in llmctl before starting the imported one, or both will use memory.</p>
    </div>
  )
}
