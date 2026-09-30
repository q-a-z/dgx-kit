import { useEffect, useState } from 'react'
import type { Action, Download, Snapshot } from '../api'
import type { Mark } from '../components/Chart'
import type { Tab } from '../components/ModelActions'
import { fetching, type Entry } from '../fleet'
import type { Alert } from '../health'
import { RUN, select } from '../nav'
import { HW_TILES } from '../tiles/hardwareTiles'
import type { Layout } from '../tiles/layout'
import type { Slo } from '../tiles/modelTiles'
import { TileGrid } from '../tiles/TileGrid'
import { usePoll } from '../usePoll'
import { MemoryMap, wants } from './MemoryMap'
import { ModelList } from './ModelList'
import { ModelPanel, type Preview } from './ModelPanel'
import { RunPanel, type Ghost } from './RunPanel'
import { SparkCard } from './SparkCard'

const MARK_LABEL: Record<string, string> = { start: 'started', stop: 'stopped', edit: 'settings changed', restore: 'settings restored', template: 'template applied' }

/** Starts, stops and settings changes for one model, from the action log, to mark on its charts. */
function marksFor(log: Action[] | null, name: string): Mark[] {
  return (log ?? []).filter((a) => MARK_LABEL[a.action] && (a.detail === name || a.detail.startsWith(`${name} `) || a.detail.endsWith(` to ${name}`)))
    .map((a) => ({ t: a.t, label: MARK_LABEL[a.action] }))
}

/**
 * The one page. Two pictures of the same box, switchable: the memory map (C) or the model list (A).
 * Either way, a model's controls sit on it, its details open in a panel of fixed width beside it,
 * and the box itself stays in view.
 */
export function Home({ view, setView, latest, history, entries, orphans, alerts, loaded, readonly, layout, update, slo, setSlo, sel, onChanged }: {
  view: 'map' | 'list'; setView: (v: 'map' | 'list') => void; latest: Snapshot; history: Snapshot[]; entries: Entry[]; orphans: Download[]; alerts: Alert[]; loaded: boolean
  readonly: boolean; layout: Layout; update: (p: Partial<Layout>) => void; slo: Slo | null; setSlo: (s: Slo) => void
  sel?: string; onChanged: () => void
}) {
  const [tab, setTab] = useState<Tab>('live')
  const [preview, setPreview] = useState<Preview>(null)
  const [ghost, setGhost] = useState<Ghost>(null)
  const log = usePoll<Action[]>('/api/log', 10000)
  // With nothing picked, show what most needs looking at, else the first model that's serving.
  const shown = sel ?? alerts.find((a) => a.model)?.model ?? entries.find((e) => e.state === 'running')?.name ?? entries[0]?.name
  useEffect(() => { setTab('live'); setPreview(null) }, [shown])
  const entry = entries.find((e) => e.name === shown)
  const onOpen = (name: string, t: Tab) => { select(name); setTimeout(() => setTab(t)) }
  const free = latest.system?.memory.available_bytes ?? 0
  const reserved = [...entries.map((e) => fetching(e.state) ? e.download : undefined), ...orphans].reduce((a, d) => a + (d ? wants(d) : 0), 0)

  const panel = sel === RUN
    ? <RunPanel readonly={readonly} onGhost={setGhost} onChanged={onChanged} />
    : entry
      ? <ModelPanel key={entry.name} entry={entry} tab={tab} setTab={setTab} latest={latest} history={history} readonly={readonly}
          layout={layout} update={update} slo={slo} setSlo={setSlo} marks={marksFor(log.data, entry.name)}
          alert={alerts.find((a) => a.model === entry.name)} onChanged={onChanged} onPreview={setPreview} />
      : (
        <section className="panel">
          {!loaded ? <p className="muted">Loading…</p> : sel ? <p className="empty-note">No model called {sel} on this box.</p> : (
            <div className="empty-note">
              <p><b>No models yet.</b> A fresh box starts empty.</p>
              <p className="muted">Run one from Hugging Face, or set up one that’s already in a folder on this box.</p>
              <button className="primary" onClick={() => select(RUN)}>Run a model</button>
            </div>
          )}
        </section>
      )
  const switcher = (
    <div className="seg" role="group" aria-label="Show models as">
      <button className={view === 'map' ? 'active' : ''} aria-pressed={view === 'map'} onClick={() => setView('map')} title="Models drawn by the memory they hold">Memory map</button>
      <button className={view === 'list' ? 'active' : ''} aria-pressed={view === 'list'} onClick={() => setView('list')} title="Models in a list with details beside it">Model list</button>
    </div>
  )
  const spark = <SparkCard latest={latest} history={history} entries={entries} showMemory={view === 'list'} reserved={reserved}
    hwOpen={!!layout.hw_open} toggleHw={() => update({ hw_open: !layout.hw_open })} />

  return (
    <>
      {view === 'map' ? (
        <div className="home">
          {spark}
          {layout.hw_open && <HardwareDetails latest={latest} history={history} layout={layout} update={update} />}
          <MemoryMap switcher={switcher} latest={latest} history={history} entries={entries} orphans={orphans} alerts={alerts} readonly={readonly}
            selected={sel === RUN ? RUN : shown} preview={preview} ghost={sel === RUN ? ghost : null}
            table={!!layout.map_table} setTable={(v) => update({ map_table: v })} onOpen={onOpen} onChanged={onChanged} />
          {panel}
        </div>
      ) : (
        <div className="home">
          {spark}
          {layout.hw_open && <HardwareDetails latest={latest} history={history} layout={layout} update={update} />}
          <div className="list-view">
          <ModelList switcher={switcher} history={history} entries={entries} orphans={orphans} alerts={alerts} readonly={readonly}
            selected={sel === RUN ? RUN : shown} free={free} onOpen={onOpen} onChanged={onChanged} />
          {panel}
          </div>
        </div>
      )}
    </>
  )
}

/** Clocks, cores, disk and network explain problems; they open on request below the rest. */
function HardwareDetails({ latest, history, layout, update }: { latest: Snapshot; history: Snapshot[]; layout: Layout; update: (p: Partial<Layout>) => void }) {
  const [arrange, setArrange] = useState(false)
  return (
    <section className="hwdetails" aria-label="Hardware details">
      <div className="section-title">
        <span className="grow">Hardware details</span>
        <button className="ghost" aria-pressed={arrange} onClick={() => setArrange(!arrange)}>{arrange ? 'Done' : 'Arrange'}</button>
        <button className="ghost" onClick={() => update({ hw_open: false })}>Hide</button>
      </div>
      <TileGrid defs={HW_TILES} tiles={layout.hardware} editing={arrange} onChange={(hardware) => update({ hardware })} ctx={{ latest, history }} />
    </section>
  )
}
