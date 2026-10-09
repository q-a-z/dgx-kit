import { useState } from 'react'
import { fmtBytes, fmtNum, type Download, type Snapshot } from '../api'
import { Spark } from '../components/Chart'
import { Icon } from '../components/Icon'
import { DownloadActions, ModelActions, type Tab } from '../components/ModelActions'
import { base, ENGINES, prettyName, STATE_LABEL, STATE_TONE, stateText, stateTitle, type Entry } from '../fleet'
import { KV_WARN, type Alert } from '../health'
import { RUN, select } from '../nav'
import { recent } from '../tiles/hardwareTiles'
import { wants } from './MemoryMap'
import { TINTS } from './SparkCard'

/**
 * Concept A. Every model in one column, each row carrying its state, its numbers and its controls.
 * Picking a row opens it in the panel beside the list; the page never changes.
 */
export function ModelList({ switcher, history, entries, orphans, alerts, readonly, selected, free, onOpen, onChanged }: {
  switcher: React.ReactNode; history: Snapshot[]; entries: Entry[]; orphans: Download[]; alerts: Alert[]; readonly: boolean; selected?: string
  free: number; onOpen: (name: string, tab: Tab) => void; onChanged: () => void
}) {
  const [find, setFind] = useState('')
  const h = recent(history, 10)
  const times = h.map((s) => s.t)
  const shown = entries.filter((e) => e.name.toLowerCase().includes(find.trim().toLowerCase()))
  return (
    <nav className="rail" aria-label="Models">
      <div className="railhead">
        {switcher}
        {entries.length > 5 && <input type="search" placeholder="Find a model" value={find} onChange={(e) => setFind(e.target.value)} aria-label="Find a model" />}
        <button className={`run ${selected === RUN ? 'on' : ''}`} onClick={() => select(RUN)}><Icon name="plus" />Run a model</button>
      </div>
      {entries.length === 0 && orphans.length === 0 && (
        <p className="empty-note">No models yet. A fresh box has none: run one from Hugging Face, or set up one already in a folder here.</p>
      )}
      {shown.map((e) => {
        const l = e.live
        const on = e.state === 'running' && l
        const a = alerts.find((x) => x.model === e.name)
        const tone = STATE_TONE[e.state] ?? ''
        const engine = e.row?.engine ?? e.ext?.engine
        return (
          <div key={e.name} className={`item ${selected === e.name ? 'sel' : ''} ${a?.level ?? ''}`} style={{ ['--tint' as string]: TINTS[entries.indexOf(e) % TINTS.length] }} onClick={() => select(e.name)}>
            <div className="irow">
              <span className={`dot ${a?.level ?? tone}`} aria-hidden />
              <button className="name" aria-current={selected === e.name} onClick={(ev) => { ev.stopPropagation(); select(e.name) }} title={e.name}>{prettyName(e)}</button>
              <ModelActions entry={e} readonly={readonly} onChanged={onChanged} onOpen={(t) => onOpen(e.name, t)} />
            </div>
            {on ? (
              <div className="imeta">
                <span className="tag">{[engine && (ENGINES[engine] ?? engine), !e.managed && 'watch only'].filter(Boolean).join(' · ')}</span>
                <span><b>{fmtNum(l.decode_tps, (l.decode_tps ?? 0) < 100 ? 1 : 0)}</b> tok/s</span>
                {l.ttft_p95_s != null && <span>p95 <b>{fmtNum(l.ttft_p95_s * 1000)}</b> ms</span>}
                <span className={(l.waiting ?? 0) > 0 ? 'warn' : ''}>{(l.waiting ?? 0) > 0 && '▲ '}queue <b>{fmtNum(l.waiting ?? 0)}</b></span>
                {l.slot && <span className={l.slot === 'waiting' ? 'muted' : ''} title={l.slot === 'waiting' ? 'Frozen until its GPU slot; its cache and streams are kept' : 'Has the GPU now'}>{l.slot === 'waiting' ? '○ waiting' : '● GPU'}</span>}
                {(l.kv_used_pct ?? 0) >= KV_WARN && <span className="warn">▲ KV <b>{fmtNum(l.kv_used_pct)}</b>%</span>}
                <span className="ispark"><Spark times={times} values={h.map((s) => (s.models[e.name]?.decode_tps as number | undefined) ?? null)} height={18} label="Decode" fmt={(v) => `${fmtNum(v, 1)} tok/s`} /></span>
              </div>
            ) : e.download ? <DownloadLine d={e.download} free={free} /> : e.service ? (
              <div className="imeta">
                <span className="tag">{ENGINES[e.name] ?? 'Decision model'}</span>
                <span className={tone === 'bad' ? 'bad' : ''} title={stateTitle(e)}>{stateText(e)}</span>
                {e.service.state === 'running' && <><span>port <b>{e.service.port}</b></span><span>{e.service.device_in_use === 'cuda' ? 'GPU' : 'CPU'}</span><span>{e.service.loaded.join(' · ')}</span></>}
              </div>
            ) : (
              <div className="imeta"><span className={tone === 'bad' ? 'bad' : ''}>{STATE_LABEL[e.state] ?? e.state}</span>{e.row && <span>{base(e.row.repo)}</span>}</div>
            )}
          </div>
        )
      })}
      {orphans.map((d) => (
        <div key={d.repo} className="item">
          <div className="irow"><span className="dot pending" aria-hidden /><span className="name">{base(d.repo)}</span><span className="tag">{d.state === 'paused' ? 'paused' : 'downloading'}</span><DownloadActions d={d} readonly={readonly} onChanged={onChanged} /></div>
          <DownloadLine d={d} free={free} />
        </div>
      ))}
    </nav>
  )
}

function DownloadLine({ d, free }: { d: Download; free: number }) {
  const need = wants(d)
  return (
    <>
      <div className="bar"><div className="used" style={{ width: `${d.pct ?? 0}%` }} /></div>
      <div className="imeta">
        <span>{fmtBytes(d.bytes)} of {fmtBytes(d.total_bytes)}</span>
        <span className={need > free ? 'warn' : ''}>{need > free ? `▲ won’t fit (${fmtBytes(need - free)} short)` : `fits (${fmtBytes(free - need)} spare)`}</span>
      </div>
    </>
  )
}
