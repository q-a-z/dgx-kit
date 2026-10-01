import { useState } from 'react'
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
