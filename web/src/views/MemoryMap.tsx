import { useState } from 'react'
import { fmtBytes, fmtNum, type Download, type Snapshot } from '../api'
import { MultiSpark } from '../components/Chart'

// Decode and prefill lines: fixed colours, validated as a pair on the dark card surface.
const RATES = ['--s1', '--s4']
import { Icon } from '../components/Icon'
import { DownloadActions, ModelActions, type Tab } from '../components/ModelActions'
import { base, ENGINES, fetching, prettyName, STATE_LABEL, STATE_TONE, stateText, stateTitle, type Entry } from '../fleet'
import { KV_WARN, type Alert } from '../health'
import { RUN, select } from '../nav'
import { recent } from '../tiles/hardwareTiles'
import type { Preview } from './ModelPanel'
import type { Ghost } from './RunPanel'
import { TINTS } from './SparkCard'

const GB = 2 ** 30
const DL_OVERHEAD = 1.1

/** Memory a download will want once it runs: its weights plus working room. */
export const wants = (d: Download) => d.total_bytes * DL_OVERHEAD

/**
 * Concept C. Unified memory decides what can run, so the home picture is that memory drawn to scale:
 * one block per model, as wide as what it holds, with its name, controls and live numbers inside.
 * Free memory is a slot that shows downloads and new models before they take room.
 */
export function MemoryMap({ switcher, latest, history, entries, orphans, alerts, readonly, selected, preview, ghost, table, setTable, onOpen, onChanged }: {
  switcher: React.ReactNode; latest: Snapshot; history: Snapshot[]; entries: Entry[]; orphans: Download[]; alerts: Alert[]; readonly: boolean
  selected?: string; preview: Preview; ghost: Ghost; table: boolean; setTable: (v: boolean) => void
  onOpen: (name: string, tab: Tab) => void; onChanged: () => void
}) {
  const [recipeFilter, setRecipeFilter] = useState('')
  const m = latest.system?.memory
  if (!m) return <section className="mapwrap"><p className="muted">Memory readings aren’t available on this box.</p></section>
  const tint = (e: Entry) => TINTS[entries.indexOf(e) % TINTS.length]
  const resident = entries.filter((e) => e.memBytes != null || e.running)
  const away = entries.filter((e) => !resident.includes(e) && !fetching(e.state))
  const recipes = away.filter((e) => {
    const query = recipeFilter.trim().toLocaleLowerCase()
    if (!query) return true
    const text = [prettyName(e), e.row?.repo, e.row?.engine, e.row?.quantization, e.row?.kv_cache_dtype, e.state]
      .filter(Boolean).join(' ').toLocaleLowerCase()
    return text.includes(query)
  })
  const held = resident.reduce((a, e) => a + (e.memBytes ?? 0), 0)
  const system = Math.max(0, m.total_bytes - m.available_bytes - held)
  const grow = preview && resident.some((e) => e.name === preview.name)
    ? preview.bytes - (resident.find((e) => e.name === preview.name)!.memBytes ?? 0) : 0
  const free = Math.max(0, m.available_bytes - grow)
  const downloads = [...entries.filter((e) => fetching(e.state) && e.download).map((e) => ({ name: e.name, d: e.download!, entry: e as Entry | undefined })),
    ...orphans.map((d) => ({ name: base(d.repo) ?? d.repo, d, entry: undefined as Entry | undefined }))]
  const h = recent(history, 10)
  const times = h.map((s) => s.t)
  const alertOf = (n: string) => alerts.find((a) => a.model === n)

  return (
    <section className="mapwrap" aria-label="Unified memory">
      <div className="map-top">{switcher}<span className="lbl">Unified memory · {fmtBytes(m.total_bytes)}</span><span className="grow" /><button className="ghost" aria-pressed={table} onClick={() => setTable(!table)}>{table ? 'Show as memory map' : 'Show as process table'}</button></div>
      {table ? <MapTable latest={latest} resident={resident} tint={tint} readonly={readonly} selected={selected} onOpen={onOpen} onChanged={onChanged} /> : (
        <div className="map">
          {resident.map((e) => {
            const bytes = preview?.name === e.name ? preview.bytes : e.memBytes ?? 0
            const l = e.live
            const on = e.state === 'running' && l
            const a = alertOf(e.name)
            const tone = STATE_TONE[e.state] ?? ''
            const engine = e.row?.engine ?? e.ext?.engine ?? (e.service ? e.name : undefined)
            return (
              <div key={e.name} className={`blk ${e.service ? 'svc' : ''} ${selected === e.name ? 'sel' : ''} ${preview?.name === e.name ? 'previewing' : ''} ${a ? a.level : ''}`}
                style={{ flexGrow: Math.max(bytes / GB, 0.5), ['--tint' as string]: tint(e) }}
                onClick={() => select(e.name)}>
                <div className="bn">
                  <span className={`dot ${a?.level ?? tone}`} aria-hidden />
                  <button className="name" onClick={(ev) => { ev.stopPropagation(); select(e.name) }} aria-current={selected === e.name} title={e.name}>{prettyName(e)}</button>
                  <ModelActions entry={e} readonly={readonly} onChanged={onChanged} onOpen={(t) => onOpen(e.name, t)} />
                </div>
                {on ? (
                  <div className="bs">
                    {l.ttft_p95_s != null && <span>p95 <b>{fmtNum(l.ttft_p95_s * 1000)}</b> ms</span>}
                    <span className={(l.waiting ?? 0) > 0 ? 'warn' : ''}>{(l.waiting ?? 0) > 0 && '▲ '}queue <b>{fmtNum(l.waiting ?? 0)}</b></span>
                    {l.kv_used_pct != null && <span className={l.kv_used_pct >= KV_WARN ? 'warn' : ''}>{l.kv_used_pct >= KV_WARN && '▲ '}KV <b>{fmtNum(l.kv_used_pct)}</b>%</span>}
                    {l.slot && <span className={l.slot === 'waiting' ? 'muted' : ''} title={l.slot === 'waiting' ? 'Frozen until its GPU slot; its cache and streams are kept' : 'Has the GPU now'}>{l.slot === 'waiting' ? '○ waiting' : '● GPU'}</span>}
                  </div>
                ) : <div className="bs"><span className={tone === 'bad' ? 'bad' : ''} title={stateTitle(e)}>{stateText(e)}</span></div>}
                {on && (
                  <div className="bsparks">
                    <div className="bs-legend">
                      {([['decode_tps', 'Decode'], ['prefill_tps', 'Prefill']] as const).map(([k, label], i) => (
                        <span key={k}><i style={{ background: `var(${RATES[i]})` }} />{label} <b>{fmtNum(l[k] as number | undefined, ((l[k] as number | undefined) ?? 0) < 100 ? 1 : 0)}</b></span>
                      ))}
                      <span className="unit">tok/s</span>
                    </div>
                    <MultiSpark times={times} height={40} fmt={(v) => `${fmtNum(v, 1)} tok/s`} series={(['decode_tps', 'prefill_tps'] as const).map((k, i) => ({ label: i ? 'Prefill' : 'Decode', color: RATES[i], values: h.map((s) => (s.models[e.name]?.[k] as number | undefined) ?? null) }))} />
                  </div>
                )}
                <span className="bfoot">
                  {preview?.name === e.name ? <>after restart <b>{fmtBytes(bytes)}</b> · now {fmtBytes(e.memBytes)}</>
                    : <><b>{e.memBytes != null ? fmtBytes(e.memBytes) : 'memory not read yet'}</b><span className="rest">{engine && ` · ${ENGINES[engine] ?? engine}`}{!e.managed && ' · watch only'}</span></>}
                </span>
              </div>
            )
          })}
          <div className="blk sys" style={{ flexGrow: Math.max(system / GB, 0.5) }} title="The OS, caches and processes that aren't models">
            <span className="lbl">System and other</span><b>{fmtBytes(system)}</b>
          </div>
          <div className="blk free" style={{ flexGrow: Math.max(free / GB, 0.5) }}>
            <div className="free-head"><span className="lbl">Free</span><b>{fmtBytes(free)}</b>{grow > 0 && <span className="muted"> after the restart you’re previewing</span>}</div>
            <div className="ghosts">
              {downloads.map(({ name, d, entry }) => {
                const need = wants(d)
                const fits = need <= free
                return (
                  <div key={d.repo} className={`gblk ${fits ? '' : 'nofit'}`} style={{ flexBasis: `${Math.min(100, (100 * need) / Math.max(free, 1))}%` }}
                    onClick={() => entry && select(entry.name)}>
                    <span className="gname"><span className="dot pending" aria-hidden />{entry ? prettyName(entry) : name}
                      {entry ? <ModelActions entry={entry} readonly={readonly} onChanged={onChanged} onOpen={(t) => onOpen(entry.name, t)} /> : <DownloadActions d={d} readonly={readonly} onChanged={onChanged} />}</span>
                    <span className="lbl">{d.state === 'paused' ? 'Paused at' : 'Downloading'} {d.pct ?? 0}% · {fmtBytes(d.bytes)} of {fmtBytes(d.total_bytes)}</span>
                    <div className="bar"><div className="used" style={{ width: `${d.pct ?? 0}%` }} /></div>
                    <span className="lbl">{fits ? `needs about ${fmtBytes(need)} · fits` : `▲ needs about ${fmtBytes(need)} · won’t fit next to what’s running`}</span>
                  </div>
                )
              })}
              {ghost && (
                <div className={`gblk lookup ${ghost.fits && ghost.bytes <= free ? '' : 'nofit'}`} style={{ flexBasis: `${Math.min(100, (100 * ghost.bytes) / Math.max(free, 1))}%` }}>
                  <span className="gname">{ghost.name}</span>
                  <span className="lbl">{ghost.fits && ghost.bytes <= free ? `would take ${fmtBytes(ghost.bytes)} · fits` : `▲ needs ${fmtBytes(ghost.bytes)} · doesn’t fit`}</span>
                </div>
              )}
            </div>
            <button className={`run ${selected === RUN ? 'on' : ''}`} onClick={(ev) => { ev.stopPropagation(); select(RUN) }}><Icon name="plus" />Run a model</button>
          </div>
        </div>
      )}
      {away.length > 0 && <section className="recipe-shelf" aria-labelledby="recipe-shelf-title">
        <div className="recipe-shelf-head">
          <div>
            <h2 id="recipe-shelf-title">Saved recipes <span className="recipe-count">{away.length}</span></h2>
            <p>Models outside GPU memory. Choose one to view its settings or start it.</p>
          </div>
          <label className="recipe-search">
            <Icon name="search" size={15} />
            <input type="search" value={recipeFilter} onChange={(e) => setRecipeFilter(e.target.value)}
              placeholder="Find a model" aria-label="Find a saved recipe" />
          </label>
        </div>
        {recipes.length > 0 ? <div className="recipe-grid" role="list">
          {recipes.map((e) => {
            const r = e.row
            const tp = r?.cluster?.workers?.length ? `TP${r.cluster.workers.length + 1}` : null
            const facts: string[] = r ? [
              ENGINES[r.engine] ?? r.engine,
              r.quantization && r.quantization.toUpperCase(),
              tp,
              r.kv_cache_dtype && r.kv_cache_dtype !== 'auto' && `${r.kv_cache_dtype.toUpperCase()} KV`,
              r.max_context && `${fmtNum(r.max_context)} context`,
              r.min_concurrency > 1 && `${fmtNum(r.min_concurrency)} parallel`,
              (r.draft_repo || r.draft_method) && `Draft: ${base(r.draft_repo) ?? r.draft_method}`,
            ].filter((fact): fact is string => typeof fact === 'string' && fact.length > 0)
              : [e.ext ? (ENGINES[e.ext.engine] ?? e.ext.engine) : null].filter((fact): fact is string => typeof fact === 'string')
            return (
              <article key={e.name} role="listitem" className={`recipe-card ${selected === e.name ? 'sel' : ''}`} onClick={() => select(e.name)}>
                <div className="recipe-card-head">
                  <div className="recipe-card-title">
                    <button className="recipe-name" onClick={(ev) => { ev.stopPropagation(); select(e.name) }} title={e.name}>{prettyName(e)}</button>
                    <span className={`recipe-state ${STATE_TONE[e.state] ?? ''}`} title={stateTitle(e)}>
                      <span className={`dot ${STATE_TONE[e.state] ?? ''}`} aria-hidden />{STATE_LABEL[e.state] ?? e.state}
                    </span>
                  </div>
                  <ModelActions entry={e} readonly={readonly} onChanged={onChanged} onOpen={(t) => onOpen(e.name, t)} />
                </div>
                {facts.length > 0 && <div className="recipe-facts">{facts.map((fact) => <span key={fact}>{fact}</span>)}</div>}
              </article>
            )
          })}
        </div> : <p className="recipe-empty">No saved models match “{recipeFilter}”.</p>}
      </section>}
    </section>
  )
}

/** The same memory as a table: one row per model with its process, for reading exact numbers. */
function MapTable({ latest, resident, tint, readonly, selected, onOpen, onChanged }: {
  latest: Snapshot; resident: Entry[]; tint: (e: Entry) => string; readonly: boolean; selected?: string
  onOpen: (name: string, tab: Tab) => void; onChanged: () => void
}) {
  const procs = latest.gpu?.processes ?? []
  return (
    <table className="maptable">
      <thead><tr><th>Model</th><th /><th>State</th><th className="num">Speed</th><th className="num">Context</th><th className="num">PID</th><th className="num">Memory</th></tr></thead>
      <tbody>
        {resident.map((e) => {
          const p = procs.find((x) => x.key === e.name || x.model === e.name)
          const decodeNow = typeof e.live?.decode_tps === 'number' ? e.live.decode_tps : 0
          const decodeRequest = typeof e.live?.decode_tps_req === 'number' ? e.live.decode_tps_req : null
          return (
            <tr key={e.name} className={selected === e.name ? 'sel' : ''} onClick={() => select(e.name)}>
              <td><button className="link who" onClick={() => select(e.name)}><i style={{ background: tint(e) }} />{prettyName(e)}</button></td>
              <td className="acts-cell"><ModelActions entry={e} readonly={readonly} onChanged={onChanged} onOpen={(t) => onOpen(e.name, t)} /></td>
              <td><span className={`state-word ${STATE_TONE[e.state] ?? ''}`} title={stateTitle(e)}>{stateText(e)}</span></td>
              <td className="num">{decodeNow > 0
                ? `${fmtNum(decodeNow, 1)} tok/s`
                : decodeRequest != null ? `${fmtNum(decodeRequest, 1)} tok/s req avg` : '–'}</td>
              <td className="num">{fmtNum((e.live?.context_tokens as number | undefined) ?? p?.ctx ?? null)}</td>
              <td className="num muted">{p?.pid ?? '–'}</td>
              <td className="num">{fmtBytes(e.memBytes)}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
