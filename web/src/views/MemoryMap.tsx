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
  const m = latest.system?.memory
  if (!m) return <section className="mapwrap"><p className="muted">Memory readings aren’t available on this box.</p></section>
  const tint = (e: Entry) => TINTS[entries.indexOf(e) % TINTS.length]
  const resident = entries.filter((e) => e.memBytes != null || e.running)
  const away = entries.filter((e) => !resident.includes(e) && !fetching(e.state))
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
            const engine = e.row?.engine ?? e.ext?.engine
            return (
              <div key={e.name} className={`blk ${selected === e.name ? 'sel' : ''} ${preview?.name === e.name ? 'previewing' : ''} ${a ? a.level : ''}`}
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
                  </div>
                ) : <div className="bs"><span className={tone === 'bad' ? 'bad' : ''}>{STATE_LABEL[e.state] ?? e.state}</span></div>}
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
      {away.length > 0 && <div className="map-foot">
        <span className="lbl">Not in memory</span>
        {away.map((e) => (
          <span key={e.name} className={`chip ${selected === e.name ? 'sel' : ''}`} onClick={() => select(e.name)}>
            <span className={`dot ${STATE_TONE[e.state] ?? ''}`} aria-hidden />
            <button className="name" onClick={(ev) => { ev.stopPropagation(); select(e.name) }} title={e.name}>{prettyName(e)}</button>
            <span className={`state-word ${STATE_TONE[e.state] ?? ''}`}>{STATE_LABEL[e.state] ?? e.state}</span>
            <ModelActions entry={e} readonly={readonly} onChanged={onChanged} onOpen={(t) => onOpen(e.name, t)} />
          </span>
        ))}
      </div>}
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
          return (
            <tr key={e.name} className={selected === e.name ? 'sel' : ''} onClick={() => select(e.name)}>
              <td><button className="link who" onClick={() => select(e.name)}><i style={{ background: tint(e) }} />{prettyName(e)}</button></td>
              <td className="acts-cell"><ModelActions entry={e} readonly={readonly} onChanged={onChanged} onOpen={(t) => onOpen(e.name, t)} /></td>
              <td><span className={`state-word ${STATE_TONE[e.state] ?? ''}`} title={stateTitle(e)}>{stateText(e)}</span></td>
              <td className="num">{e.live?.decode_tps != null ? `${fmtNum(e.live.decode_tps, 1)} tok/s` : '–'}</td>
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
