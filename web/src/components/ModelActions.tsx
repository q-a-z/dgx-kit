import { useState } from 'react'
import { api, type Download } from '../api'
import { useModelActions } from '../actions'
import { fetching, type Entry } from '../fleet'
import { Icon } from './Icon'

export type Tab = 'live' | 'settings' | 'logs' | 'bench'

/**
 * The controls for one model, drawn wherever the model's name is: on its block, its list row, its panel.
 * Only the actions that make sense in its current state appear.
 */
export function ModelActions({ entry, readonly, onChanged, onOpen, labels = false }: {
  entry: Entry; readonly: boolean; onChanged: () => void; onOpen: (tab: Tab) => void; labels?: boolean
}) {
  const a = useModelActions(entry, readonly, onChanged)
  const s = entry.state
  const off = a.locked ? a.why ?? `${a.busy}…` : undefined
  const btn = (icon: string, label: string, run: () => unknown, o: { disabled?: boolean; primary?: boolean; hint?: string } = {}) => (
    <button key={label} className={`${labels ? 'act' : 'ic'} ${o.primary ? 'primary' : ''}`} disabled={o.disabled}
      title={o.disabled ? o.hint ?? off : label} aria-label={`${label} ${entry.name}`}
      onClick={(e) => { e.stopPropagation(); run() }}>
      <Icon name={icon} />{labels && <span>{label}</span>}
    </button>
  )
  const control = { disabled: a.locked, hint: off }
  const items = [
    ...(entry.running ? [btn('restart', a.busy === 'Starting' ? 'Starting…' : 'Restart', a.restart, control), btn('stop', 'Stop', a.stop, control)] : []),
    // A crashed or crash-looping model still has a container: Stop clears it and frees its port.
    ...(!entry.running && entry.managed && ['exited', 'restarting', 'starting'].includes(s) && entry.row?.container ? [btn('stop', 'Stop', a.stop, control)] : []),
    ...(['stopped', 'exited'].includes(s) ? [btn('start', a.busy ? `${a.busy}…` : 'Start', a.start, { ...control, primary: labels })] : []),
    ...(s === 'missing' || s === 'failed' ? [btn('download', s === 'failed' ? 'Retry download' : 'Download', a.download, { ...control, primary: labels })] : []),
    ...(s === 'downloading' ? [btn('pause', 'Pause download', a.pause, control), btn('cancel', 'Cancel download', a.cancel, control)] : []),
    ...(s === 'paused' ? [btn('start', 'Resume download', a.resume, { ...control, primary: labels }), btn('cancel', 'Cancel download', a.cancel, control)] : []),
    ...(fetching(s) ? [] : [btn('logs', 'Logs', () => onOpen('logs'))]),
    ...(labels && entry.managed ? [btn('settings', 'Settings', () => onOpen('settings'))] : []),
    // Whether clients can reach it through the gateway; it keeps running either way, and a stopped model publishes once it runs.
    ...(labels && entry.managed && entry.row
      ? [btn(entry.row.publish ? 'unpublish' : 'publish', entry.row.publish ? 'Unpublish' : 'Publish', () => a.publish(!entry.row!.publish), control)]
      : []),
  ]
  return (
    <span className={`acts ${labels ? 'labelled' : ''}`} onClick={(e) => e.stopPropagation()}>
      {items}
      {a.msg && (
        <span className={`act-msg ${a.msg.bad ? 'bad' : ''}`} role="alert">
          {a.msg.text}
          <button className="ic" aria-label="Dismiss" onClick={() => a.setMsg(null)}><Icon name="close" size={12} /></button>
        </span>
      )}
    </span>
  )
}

/** Pause, resume and cancel for a download that has no model entry yet (a draft on its own, say). */
export function DownloadActions({ d, readonly, onChanged }: { d: Download; readonly: boolean; onChanged: () => void }) {
  const [err, setErr] = useState<string | null>(null)
  const run = (path: string, method = 'POST') => api(path, { method }).then(() => setErr(null), (e: Error) => setErr(e.message)).finally(onChanged)
  const hint = readonly ? 'Read-only: DGX-kit won’t download anything on this box.' : undefined
  const b = (icon: string, label: string, path: string, method?: string) => (
    <button key={label} className="ic" disabled={readonly} title={hint ?? label} aria-label={`${label} ${d.repo}`}
      onClick={(e) => { e.stopPropagation(); run(path, method) }}><Icon name={icon} /></button>
  )
  return (
    <span className="acts" onClick={(e) => e.stopPropagation()}>
      {d.state === 'paused' ? b('start', 'Resume download', `/api/downloads/${d.repo}/resume`) : d.state !== 'failed' && b('pause', 'Pause download', `/api/downloads/${d.repo}/pause`)}
      {b('cancel', 'Cancel download', `/api/downloads/${d.repo}`, 'DELETE')}
      {err && <span className="act-msg bad" role="alert">{err}</span>}
    </span>
  )
}
