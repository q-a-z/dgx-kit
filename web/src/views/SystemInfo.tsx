import { useEffect, useState } from 'react'
import { useConfirm } from '../components/Confirm'
import { toast } from '../toast'
import { api } from '../api'
import { usePoll } from '../usePoll'
import { Pending } from './Settings'

type Device = { id: string; name: string; version: string | null; vendor: string | null; summary: string | null; updatable: boolean; updates: { version: string | null; summary: string | null }[] }
type Report = {
  items: { id: string; group: string; label: string; value: string }[]
  firmware: { checked: number | null; error: string | null; devices: Device[]; updates: number }
  changes: { t: number; key: string; label: string; from: string | null; to: string }[]
  since: number | null
}

const when = (t: number) => new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })

export type UpdateState = {
  current: string
  latest: { version?: string; sha?: string; notes?: string; checked?: number; error?: string; repo?: string } | null
  available: boolean
  job: { version: string | null; state: string; step: string; error: string | null; tail: string[] } | null
}

/** Update DGX-kit from GitHub: what is out there, what's new, and one button. Models keep running during an update. */
function UpdateCard() {
  const ask = useConfirm()
  const [fast, setFast] = useState(false)
  const { data, reload } = usePoll<UpdateState>('/api/system/update', fast ? 2000 : 60000)
  const [busy, setBusy] = useState(false)
  const job = data?.job
  const working = job?.state === 'running' || job?.state === 'restarting'
  if (working !== fast) setFast(working)
  // After the restart the page that asked is talking to the new dashboard: reload when its version changes.
  useEffect(() => {
    if (job?.state !== 'restarting' || !data) return
    const id = window.setInterval(() => {
      fetch('/api/me').then((r) => r.json()).then((m: { version?: string }) => { if (m.version && m.version !== data.current) location.reload() }).catch(() => {})
    }, 2000)
    return () => window.clearInterval(id)
  }, [job?.state, data?.current]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!data) return null
  const latest = data.latest
  const check = () => { setBusy(true); api('/api/system/update/check', { method: 'POST' }).then(reload).catch((e: Error) => toast(e.message, true)).finally(() => setBusy(false)) }
  const update = async () => {
    const ok = await ask({
      title: data.available ? `Update to ${latest?.version}?` : 'Reinstall the latest from GitHub?',
      body: <>DGX-kit downloads the latest source from GitHub, builds it and restarts the dashboard on it. That takes a few minutes, and the page is unavailable for a moment at the end. Your models keep running, and your settings, keys and recipes are untouched. The current version stays as a rollback image.</>,
      label: 'Update',
    })
    if (!ok) return
    api('/api/system/update', { method: 'POST' }).then(() => { toast('Update started.'); reload() }).catch((e: Error) => toast(e.message, true))
  }
  return (
    <section className="card wide">
      <div className="row">
        <h2 className="grow">DGX-kit update</h2>
        <button disabled={busy || working} onClick={check}>{busy ? 'Checking…' : 'Check now'}</button>
        <button className={data.available ? 'primary' : ''} disabled={working || !latest?.version} onClick={update}>{data.available ? `Update to ${latest?.version}` : 'Reinstall latest'}</button>
      </div>
      <p className={data.available ? 'warn' : 'muted'}>
        {latest?.error ? latest.error
          : !latest?.version ? 'Not checked yet.'
          : data.available ? `▲ Version ${latest.version} is available${latest.sha ? ` (commit ${latest.sha})` : ''}. You have ${data.current}.`
          : `You have ${data.current}, the latest on GitHub.`}
      </p>
      {working && job && (
        <div className="pull-progress">
          <progress className="pull" max={1} aria-label="update progress" />
          <p className="muted">{job.step}{job.state === 'restarting' ? '… the page reloads by itself when the new version is up.' : '…'}</p>
          <pre className="logs">{job.tail.join('\n')}</pre>
        </div>
      )}
      {job?.state === 'failed' && <p className="bad">The update failed: {job.error}</p>}
      {data.available && latest?.notes && <details className="more"><summary>What’s new</summary><pre className="logs">{latest.notes}</pre></details>}
      <p className="muted small">From {latest?.repo ?? 'GitHub'}. The same as running <code>dgx-kit update</code> on the machine. To go back: <code>docker tag dgx-kit:previous dgx-kit:latest && docker stop dgx-kit</code>.</p>
    </section>
  )
}

/** What the machine is made of, its firmware and whether fwupd knows of updates, and what changed since DGX-kit started watching. */
export function SystemInfo() {
  const { data, error, reload } = usePoll<Report>('/api/system', 60000)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  if (!data) return <Pending error={error} what="the system information" />
  const fw = data.firmware
  const check = () => {
    setBusy(true); setMsg(null)
    api('/api/system/firmware/check', { method: 'POST' }).then(reload).catch((e: Error) => setMsg(e.message)).finally(() => setBusy(false))
  }
  const groups = ['Machine', 'Firmware', 'Software'].map((g) => ({ g, rows: data.items.filter((i) => i.group === g) })).filter((x) => x.rows.length)
  const updatable = fw.devices.filter((d) => d.updatable)
  const other = fw.devices.filter((d) => !d.updatable)
  return (
    <>
      <UpdateCard />
      <section className="card wide">
        <div className="row">
          <h2 className="grow">Firmware updates</h2>
          <button className="primary" disabled={busy} onClick={check}>{busy ? 'Checking…' : 'Check for updates'}</button>
        </div>
        <p className={fw.updates ? 'warn' : 'muted'}>
          {fw.checked == null ? 'Not checked yet. DGX-kit also checks once a day.'
            : fw.updates ? `▲ ${fw.updates} firmware update${fw.updates === 1 ? '' : 's'} available.`
            : 'Everything fwupd manages is up to date.'}
          {fw.checked != null && <span className="muted"> Last checked {when(fw.checked)}.</span>}
        </p>
        {(fw.error || msg) && <p className="bad">{msg ?? `The last check failed: ${fw.error}`}</p>}
        {updatable.length > 0 && (
          <table><tbody>
            {updatable.map((d) => (
              <tr key={d.id}>
                <td>{d.name}<div className="muted small">{[d.vendor, d.summary].filter(Boolean).join(' · ')}</div></td>
                <td className="num">{d.version ?? '–'}</td>
                <td>{d.updates.length ? <span className="warn">▲ {d.updates[0].version} available{d.updates[0].summary ? `: ${d.updates[0].summary}` : ''}</span> : <span className="muted">up to date</span>}</td>
              </tr>
            ))}
          </tbody></table>
        )}
        {other.length > 0 && (
          <details className="more">
            <summary>Keys, certificates and other read-only entries ({other.length})</summary>
            <table><tbody>{other.map((d) => <tr key={d.id}><td>{d.name}</td><td className="num">{d.version ?? '–'}</td></tr>)}</tbody></table>
          </details>
        )}
        <p className="muted small">DGX-kit only reports. To install an update, run <code>sudo fwupdmgr update</code> on the machine and reboot when it asks.</p>
      </section>

      {groups.map(({ g, rows }) => (
        <section key={g} className="card wide">
          <h3>{g}</h3>
          <table><tbody>{rows.map((r) => <tr key={r.id}><td className="muted">{r.label}</td><td className="num">{r.value}</td></tr>)}</tbody></table>
        </section>
      ))}

      <section className="card wide">
        <h3>Changes</h3>
        {data.changes.length ? (
          <table><tbody>
            {data.changes.map((c, i) => (
              <tr key={i}><td className="muted">{when(c.t)}</td><td>{c.label}</td><td className="num">{c.from} → {c.to}</td></tr>
            ))}
          </tbody></table>
        ) : <p className="muted">No change yet{data.since ? ` since DGX-kit started watching on ${when(data.since)}` : ''}. A new BIOS, driver, kernel or firmware version will show up here with its date.</p>}
      </section>
    </>
  )
}
