import { Pending } from './Settings'
import { toast } from '../toast'
import { PullProgress } from './Other'
import { useState } from 'react'
import { api, copyText, fmtBytes, fmtNum, fmtRate, type Download, type Gateway, type ModelRow, type Plan, type Recipe } from '../api'
import { usePoll } from '../usePoll'
import { Stat } from '../components/Stat'

function PlanLine({ plan }: { plan: Plan }) {
  return (
    <p className={plan.fits ? 'muted' : 'bad'}>
      {plan.fits
        ? `Fits: ${fmtNum(plan.context_tokens)} context, ${fmtNum(plan.kv_pool_tokens)} KV tokens (${fmtBytes(plan.kv_bytes)}), about ${fmtNum(plan.concurrency, 1)} requests at once. About ${fmtBytes(plan.total_bytes ?? 0)} of memory in all.`
        : `Doesn't fit: ${plan.reason}`}
    </p>
  )
}

const eta = (d: Download) => {
  if (!d.bytes_per_s || !d.total_bytes) return null
  const s = Math.max(0, (d.total_bytes - d.bytes) / d.bytes_per_s)
  return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`
}

/** One download's progress, shown on the model it belongs to while it runs. */
export function DownloadProgress({ d, onChange }: { d: Download; onChange: () => void }) {
  const left = d.state === 'running' ? eta(d) : null
  return (
    <div className="dl">
      <div className="row">
        <span className="grow">{d.state === 'failed' ? 'Download failed' : d.state === 'queued' ? 'Waiting to download' : 'Downloading'} <code>{d.repo}</code></span>
        <span className="muted">
          {fmtBytes(d.bytes)} of {fmtBytes(d.total_bytes)}{d.pct != null ? ` (${d.pct}%)` : ''}
          {d.state === 'running' ? ` · ${fmtRate(d.bytes_per_s)}` : ''}{left ? ` · ${left} left` : ''}
        </span>
        {d.state !== 'failed' && <button onClick={() => api(`/api/downloads/${d.repo}`, { method: 'DELETE' }).then(onChange).catch(() => {})}>Cancel</button>}
      </div>
      <div className="bar"><div className="used" style={{ width: `${d.pct ?? 0}%` }} /></div>
      {d.error && <p className="bad">{d.error}</p>}
    </div>
  )
}

/** Startup settings as an editable form; unknown keys stay in the recipe untouched. */
type Template = { name: string; builtin: boolean; about: string }

function TemplateBar({ model, onDone }: { model: ModelRow; onDone: (msg: string) => void }) {
  const { data } = usePoll<Template[]>('/api/templates', 15000)
  const [pick, setPick] = useState('')
  const [saveAs, setSaveAs] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const chosen = data?.find((t) => t.name === pick)
  const applyIt = () =>
    api<{ plan: Plan | null }>(`/api/models/${model.name}/apply-template/${pick}`, { method: 'POST' })
      .then((r) => onDone(`Applied ${pick}.${r.plan ? (r.plan.fits ? ` It fits with ${fmtNum(r.plan.context_tokens)} context.` : ` It doesn't fit: ${r.plan.reason}`) : ''}`))
      .catch((e: Error) => setErr(e.message))
  const save = () =>
    api(`/api/models/${model.name}/save-template`, { method: 'POST', json: { name: saveAs.trim() } })
      .then(() => onDone(`Saved these settings as template ${saveAs.trim()}.`))
      .catch((e: Error) => setErr(e.message))
  return (
    <div className="row full">
      <select value={pick} onChange={(e) => setPick(e.target.value)} style={{ width: 'auto' }}>
        <option value="">Template…</option>
        {data?.map((t) => <option key={t.name} value={t.name}>{t.name}{t.builtin ? '' : ' (yours)'}</option>)}
      </select>
      <button disabled={!pick} onClick={applyIt}>Apply</button>
      {chosen && <span className="muted">{chosen.about}</span>}
      <span className="grow" />
      <input style={{ width: 180 }} placeholder="template name" value={saveAs} onChange={(e) => setSaveAs(e.target.value)} />
      <button disabled={!saveAs.trim()} onClick={save}>Save as template</button>
      {err && <p className="bad full">{err}</p>}
    </div>
  )
}

/** The image this one model runs: the engine's default, a local build (such as patched vLLM 0.29), or any tag. */
function ImagePicker({ engine, value, onChange }: { engine: string; value: string | null; onChange: (v: string | null) => void }) {
  const { data } = usePoll<{ default: string; choices: string[] }>(`/api/images/choices/${engine}`, 15000)
  const listed = !value || !!data?.choices.includes(value)
  const [other, setOther] = useState(false)
  const status = usePoll<{ ready: boolean; build: string | null; job: { kind: string; state: string; error: string | null; tail: string[]; progress?: number | null } | null }>(`/api/images/status?image=${encodeURIComponent(value ?? '')}`, 2000, !!value)
  const [pullErr, setPullErr] = useState<string | null>(null)
  const pull = () => { setPullErr(null); (st?.build ? api(`/api/images/builds/${st.build}`, { method: 'POST' }) : api('/api/images/pull', { method: 'POST', json: { image: value } })).then(status.reload).catch((e: Error) => setPullErr(e.message)) }
  const st = status.data
  const pulling = st?.job?.state === 'running'
  return (
    <label>Image
      {other ? (
        <span className="row">
          <input className="grow" value={value ?? ''} placeholder="registry/name:tag" onChange={(e) => onChange(e.target.value.trim() || null)} />
          <button type="button" onClick={() => { setOther(false); onChange(null) }}>List</button>
        </span>
      ) : (
        <select value={value ?? ''} onChange={(e) => (e.target.value === '\u0000' ? setOther(true) : onChange(e.target.value || null))}>
          <option value="">Engine default{data ? ` (${data.default})` : ''}</option>
          {data?.choices.filter((c) => c !== data.default).map((c) => <option key={c} value={c}>{c}</option>)}
          {value && !listed && <option value={value}>{value}</option>}
          <option value={'\u0000'}>Other tag…</option>
        </select>
      )}
      {value && st && (
        <small className={pulling || st.ready ? 'muted' : 'warn'}>
          {pulling ? (st.job!.kind === 'build' ? 'Building…' : 'Pulling…')
            : st.ready ? 'On this box.'
            : st.job?.state === 'failed' ? `${st.build ? 'Build' : 'Pull'} failed: ${st.job.error ?? ''}` : st.build ? 'Not built on this box yet; the model can’t start until it is built.' : 'Not on this box yet; the model can’t start until it is pulled.'}
          {!pulling && !st.ready && <> <button type="button" onClick={pull}>{st.job?.state === 'failed' ? 'Try again' : st.build ? 'Build' : 'Pull'}</button></>}
          {pullErr && <span className="bad"> {pullErr}</span>}
        </small>
      )}
      {value && st?.job && st.job.state === 'running' && (
        <PullProgress job={st.job} />
      )}
    </label>
  )
}

export function Editor({ model, onDone }: { model: ModelRow; onDone: (msg: string) => void }) {
  const [r, setR] = useState<Recipe>(model)
  const [extra, setExtra] = useState(model.extra_args.join('\n'))
  const [options, setOptions] = useState(model.options ?? '')
  const [versions, setVersions] = useState<string[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const set = <K extends keyof Recipe>(k: K, v: Recipe[K]) => setR({ ...r, [k]: v })
  const asText = r.engine === 'vllm' && model.options != null  // one text box for vLLM; the fields stay for other engines
  const numOrNull = (s: string) => (s.trim() === '' ? null : Number(s))

  const save = async () => {
    try {
      const { container: _c, live: _l, downloaded: _d, ...body } = { ...model, ...r } as ModelRow
      const res = await api<{ applies_on_next_start: boolean }>(`/api/models/${model.name}`, {
        method: 'PUT',
        json: asText ? { ...body, options } : { ...body, extra_args: extra.split('\n').map((s) => s.trim()).filter(Boolean) },
      })
      onDone(res.applies_on_next_start ? 'Saved. Restart the model to apply.' : 'Saved.')
    } catch (e) {
      setErr((e as Error).message)
    }
  }
  const restore = async (v: string) => {
    await api(`/api/models/${model.name}/restore/${v}`, { method: 'POST' })
    onDone(`Restored version ${v}.`)
  }

  return (
    <div className="editor">
      <TemplateBar model={model} onDone={onDone} />
      <label>Engine
        <select value={r.engine} onChange={(e) => set('engine', e.target.value)}>
          <option value="vllm">vLLM</option><option value="sglang">SGLang</option><option value="llamacpp">llama.cpp</option>
        </select>
      </label>
      <ImagePicker engine={r.engine} value={r.image ?? null} onChange={(v) => set('image', v)} />
      {!asText && <>
      <label>Quantization<input value={r.quantization ?? ''} onChange={(e) => set('quantization', e.target.value || null)} /></label>
      <label>KV cache type
        <select value={r.kv_cache_dtype} onChange={(e) => set('kv_cache_dtype', e.target.value)}>
          <option value="auto">auto</option><option value="fp8">fp8</option><option value="bf16">bf16</option>
        </select>
      </label>
      <label>Max context<input type="number" value={r.max_context ?? ''} placeholder="from model" onChange={(e) => set('max_context', numOrNull(e.target.value))} /></label>
      <label>Min context<input type="number" value={r.min_context} onChange={(e) => set('min_context', Number(e.target.value))} /></label>
      <label>Min concurrent requests<input type="number" step="0.5" value={r.min_concurrency} onChange={(e) => set('min_concurrency', Number(e.target.value))} /></label>
      <label>Draft model<input value={r.draft_repo ?? ''} placeholder="none" onChange={(e) => set('draft_repo', e.target.value || null)} /></label>
      <label>Draft method<input value={r.draft_method ?? ''} onChange={(e) => set('draft_method', e.target.value || null)} /></label>
      <label>Speculative tokens<input type="number" value={r.num_speculative_tokens} onChange={(e) => set('num_speculative_tokens', Number(e.target.value))} /></label>
      {r.engine === 'llamacpp' && <label>GGUF file<input value={r.gguf_file ?? ''} onChange={(e) => set('gguf_file', e.target.value || null)} /></label>}
      </>}
      <label className="check full">
        <input type="checkbox" checked={!!r.fill_memory} onChange={(e) => set('fill_memory', e.target.checked)} />
        Use all free memory (longest context that fits, every free byte for the KV cache). Off: the model takes only what its context needs.
      </label>
      <label className="check full">
        <input type="checkbox" checked={r.publish} onChange={(e) => set('publish', e.target.checked)} />
        Publish on the gateway while it runs
      </label>
      {asText
        ? <label className="full">Engine options, one flag with its value per line
            <textarea className="mono" rows={20} spellCheck={false} value={options} onChange={(e) => setOptions(e.target.value)} />
            <small className="muted">Same as an llmctl .conf: remove a line to clear it. Without --max-model-len and --kv-cache-memory-bytes DGX-kit sizes them from free memory.</small>
          </label>
        : <label className="full">Extra engine flags, one per line, each flag with its value (e.g. --max-num-seqs 2)<textarea rows={12} value={extra} onChange={(e) => setExtra(e.target.value)} /></label>}
      {err && <p className="bad full">{err}</p>}
      <div className="row full">
        <button className="primary" onClick={save}>Save settings</button>
        <button onClick={() => api<string[]>(`/api/models/${model.name}/versions`).then(setVersions)}>Earlier versions</button>
        {versions && (versions.length === 0
          ? <span className="muted">None yet</span>
          : versions.map((v) => <button key={v} onClick={() => restore(v)}>Restore {v}</button>))}
      </div>
    </div>
  )
}

type ExtraEnvState = { file: string; text: string; applied: string[]; ignored: { line: number; text: string; why: string }[]; pending: boolean }

/** LiteLLM's own environment variables (litellm/extra.env in the state folder), edited here. */
function ExtraEnv({ onApplied }: { onApplied: () => void }) {
  const { data, reload } = usePoll<ExtraEnvState>('/api/gateway/extra', 30000)
  const [text, setText] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null)
  if (!data) return null
  const shown = text ?? data.text
  const put = (apply: boolean) => {
    if (apply && !window.confirm('Apply now? The LiteLLM gateway restarts for a few seconds; running models are not touched.')) return
    setBusy(true)
    api<ExtraEnvState>('/api/gateway/extra', { method: 'PUT', json: { text: shown, apply } })
      .then(() => { setText(null); setMsg({ text: apply ? 'Saved and applied.' : 'Saved. Not applied yet.' }); toast(apply ? 'LiteLLM settings saved and applied.' : 'LiteLLM settings saved. Not applied yet.'); reload(); onApplied() })
      .catch((e: Error) => setMsg({ text: e.message, bad: true }))
      .finally(() => setBusy(false))
  }
  return (
    <label>Extra LiteLLM settings
      <textarea className="mono" rows={6} spellCheck={false} value={shown} onChange={(e) => setText(e.target.value)} />
      <small className="muted">
        One <code>NAME=value</code> per line, for example <code>STORE_MODEL_IN_DB=True</code>; <code>#</code> starts a comment. File: <code>{data.file}</code>.
        {data.applied.length ? ` Saved: ${data.applied.join(', ')}.` : ' Nothing set.'}
        {data.pending && text === null && ' Saved but not applied to the running gateway yet.'}
      </small>
      {data.ignored.map((i) => <small key={i.line} className="bad">Line {i.line} ignored ({i.why}): {i.text}</small>)}
      <span className="row">
        <button disabled={busy || text === null} onClick={() => put(false)}>Save</button>
        <button className="primary" disabled={busy || (text === null && !data.pending)} onClick={() => put(true)}>Save and apply</button>
        {msg && <span className={msg.bad ? 'bad' : 'muted'}>{msg.text}</span>}
      </span>
    </label>
  )
}

type GatewayConf = { url: string | null; url_default: string; key_set: boolean; key_hint: string | null }

/** Where the gateway is and its key: DGX-kit publishes running models there and checks it with them. */
export function GatewayCard() {
  const { data, reload } = usePoll<Gateway>('/api/gateway', 5000)
  const conf = usePoll<GatewayConf>('/api/settings/gateway', 60000)
  const [url, setUrl] = useState<string | null>(null)
  const [key, setKey] = useState('')
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [setting, setSetting] = useState(false)
  if (!conf.data) return <Pending error={conf.error} what="the gateway settings" />
  const c = conf.data
  const shownUrl = url ?? c.url ?? ''
  const save = (body: object) => api<GatewayConf>('/api/settings/gateway', { method: 'PUT', json: body })
    .then(() => { setUrl(null); setKey(''); setMsg({ text: 'Saved.' }); toast('Gateway settings saved.'); conf.reload(); reload() })
    .catch((e: Error) => setMsg({ text: e.message, bad: true }))
  const copy = async (what: string, get: () => Promise<string>) => {
    const ok = await copyText(await get())
    setCopied(ok ? what : `${what}-failed`); setTimeout(() => setCopied(null), 1500)
  }
  const status = !data ? null
    : data.reachable === false ? { bad: !data.problem?.startsWith('pulling'), text: data.problem ?? `Not answering at ${data.url}${data.error ? `: ${data.error}` : ''}` }
    : data.auth === 'rejected' ? { bad: true, text: 'Answering, but it rejects this key. Models can’t be published until the key is right.' }
    : data.reachable ? { bad: false, text: `Answering${data.external ? ` (LiteLLM already on this box: ${data.external})` : ''}. Serves ${data.served?.length ? data.served.join(', ') : 'no models yet'}.` }
    : null
  return (
    <section className="card wide gateway-settings">
      {status && <p className={status.bad ? 'bad' : ''}><span aria-hidden>{status.bad ? '▲ ' : '● '}</span>{status.text}</p>}
      {data && (
        <p className="gw-row">
          <code>{(data.url ?? `${location.hostname}:${data.port}/v1`).replace(/^https?:\/\//, '')}</code>
          <a className="gw-link" href={(data.url ?? `${location.protocol}//${location.hostname}:${data.port}/v1`).replace(/\/v1$/, '/ui/')} target="_blank" rel="noreferrer">Open LiteLLM ↗</a>
        </p>
      )}
      {data && !data.external && !c.url && (
        <>
          <ul className="muted small">
            <li>Master key: {data.key_ready ? 'generated' : 'missing'}</li>
            <li>Database (Postgres): {data.db ?? 'not used'}</li>
            <li>LiteLLM: {data.state ?? 'not created'}</li>
          </ul>
          <div className="row">
            <button className={data.reachable ? '' : 'primary'} disabled={setting}
              onClick={() => { setSetting(true); api('/api/gateway/sync', { method: 'POST' }).then(reload).catch((e: Error) => setMsg({ text: e.message, bad: true })).finally(() => setSetting(false)) }}>
              {setting ? 'Setting up…' : data.reachable ? 'Re-run LiteLLM setup' : 'Set up LiteLLM'}
            </button>
            <span className="muted">Makes the key, pulls the images, starts Postgres and LiteLLM. Safe to run again; it only repairs what’s missing.</span>
          </div>
          {data.extra_env_file && <ExtraEnv onApplied={reload} />}
        </>
      )}
      <label>Address clients use
        <span className="row">
          <input className="grow" value={shownUrl} placeholder={c.url_default} onChange={(e) => setUrl(e.target.value)} />
          <button onClick={() => copy('url', async () => data?.url ?? c.url ?? c.url_default)}>{copied === 'url' ? 'Copied' : copied === 'url-failed' ? 'Copy failed' : 'Copy'}</button>
        </span>
        <small className="muted">Leave empty to use {c.url_default}. When set, DGX-kit publishes running models to this LiteLLM through its API and adds or removes only its own entries.</small>
      </label>
      <label>Key (LiteLLM master key)
        <span className="row">
          <input className="grow" type="password" autoComplete="off" value={key} placeholder={c.key_set ? `saved ${c.key_hint ?? ''}` : 'not set'} onChange={(e) => setKey(e.target.value)} />
          {c.key_set && <button onClick={() => copy('key', () => api<{ key: string }>('/api/settings/gateway/key').then((r) => r.key))}>{copied === 'key' ? 'Copied' : copied === 'key-failed' ? 'Copy failed' : 'Copy'}</button>}
          {c.key_set && <button className="ghost" onClick={() => save({ clear_key: true })}>Remove</button>}
        </span>
      </label>
      <div className="row">
        <button className="primary" disabled={url === null && !key} onClick={() => save({ ...(url !== null ? { url } : {}), ...(key ? { key } : {}) })}>Save</button>
        {msg && <span className={msg.bad ? 'bad' : 'muted'}>{msg.text}</span>}
      </div>
    </section>
  )
}

export function AddModel({ onAdded, onFound }: { onAdded: (name: string) => void; onFound?: (r: Recipe | null, p: Plan | null) => void }) {
  const [repo, setRepo] = useState('')
  const [draft, setDraft] = useState('')
  const [found, setFound] = useState<{ recipe: Recipe; plan: Plan | null } | null>(null)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const inspect = async () => {
    setBusy(true)
    setErr(null)
    try {
      const res = await api<{ recipe: Recipe; plan: Plan | null }>('/api/recipes/inspect', {
        method: 'POST',
        json: { repo: repo.trim(), draft_repo: draft.trim() || null },
      })
      setFound(res)
      setName(res.recipe.name)
      onFound?.(res.recipe, res.plan)
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const add = async (download: boolean) => {
    if (!found) return
    try {
      await api('/api/models', { method: 'POST', json: { ...found.recipe, name } })
      if (download) await api(`/api/models/${name}/download`, { method: 'POST' })
      setFound(null)
      onFound?.(null, null)
      setRepo('')
      setDraft('')
      onAdded(name)
    } catch (e) {
      setErr((e as Error).message)
    }
  }

  return (
    <div className="add-model">
      <div className="row">
        <input className="grow" placeholder="Model id, e.g. Qwen/Qwen3-8B" value={repo} onChange={(e) => setRepo(e.target.value)} />
        <input className="grow" placeholder="Draft model id (optional)" value={draft} onChange={(e) => setDraft(e.target.value)} />
        <button className="primary" disabled={!repo.trim() || busy} onClick={inspect}>{busy ? 'Reading…' : 'Look up'}</button>
      </div>
      {err && <p className="bad">{err}</p>}
      {found && (
        <div className="found">
          <div className="stats">
            <Stat label="Engine" value={found.recipe.engine} />
            <Stat label="Quantization" value={found.recipe.quantization ?? 'none'} />
            <Stat label="Weights" value={fmtBytes(found.recipe.weights_bytes + found.recipe.draft_weights_bytes)} />
            {found.recipe.draft_method && <Stat label="Draft" value={found.recipe.draft_method} />}
          </div>
          {found.plan ? <PlanLine plan={found.plan} /> : <p className="muted">Pick a GGUF file in the settings after adding to size this one.</p>}
          <div className="row">
            <label>Name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
            <button className="primary" onClick={() => add(true)}>Download</button>
            <button onClick={() => add(false)}>Add without downloading</button>
          </div>
        </div>
      )}
    </div>
  )
}
