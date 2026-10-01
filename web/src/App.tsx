import { useEffect, useState } from 'react'
import { api, type Gateway } from './api'
import { useFleet, type Entry } from './fleet'
import { alertsFor, type Alert } from './health'
import { Login } from './Login'
import { href, select, useRoute } from './nav'
import { HW_DEFAULTS } from './tiles/hardwareTiles'
import { useLayout, type Layout } from './tiles/layout'
import { MODEL_DEFAULTS, type Slo } from './tiles/modelTiles'
import { PANEL_DEFAULTS } from './tiles/panelTiles'
import { usePoll } from './usePoll'
import { useStream } from './useStream'
import { Home } from './views/Home'
import { Settings } from './views/Settings'
import { LibraryPage } from './views/Library'

type Me = { auth_required: boolean; signed_in: boolean; readonly?: boolean; version?: string }
const DEFAULTS: Layout = { model: MODEL_DEFAULTS, hardware: HW_DEFAULTS, panel: PANEL_DEFAULTS, stat: 'p95' }

/** Checks the session first, so the live stream only opens once signed in. */
export default function App() {
  const [me, setMe] = useState<Me | null>(null)
  const check = () => api<Me>('/api/me').then(setMe).catch(() => setMe({ auth_required: false, signed_in: true }))
  useEffect(() => {
    check()
    // Any API call that comes back 401 (session expired, password changed elsewhere) lands here.
    const onOut = () => setMe({ auth_required: true, signed_in: false })
    window.addEventListener('dgxkit:signed-out', onOut)
    return () => window.removeEventListener('dgxkit:signed-out', onOut)
  }, [])
  if (!me) return null
  if (!me.signed_in) return <Login onDone={check} version={me.version} />
  return <Dashboard version={me.version} readonly={!!me.readonly} canSignOut={me.auth_required} onSignOut={() => api('/api/logout', { method: 'POST' }).then(check)} />
}

function Dashboard({ readonly, canSignOut, onSignOut, version }: { readonly: boolean; canSignOut: boolean; onSignOut: () => void; version?: string }) {
  const { latest, history, connected } = useStream()
  const route = useRoute()
  const fleet = useFleet(latest)
  const { layout, update } = useLayout(DEFAULTS)
  const gateway = usePoll<Gateway>('/api/gateway', 10000)
  const [slo, setSlo] = useState<Slo | null>(null)
  useEffect(() => {
    api<Slo>('/api/settings/slo').then(setSlo).catch(() => {})
  }, [])
  const g = gateway.data
  const gwUp = !!g && (g.reachable ?? g.state === 'running') && g.auth !== 'rejected'
  const gwState = !g ? 'pending' : gwUp ? 'ok' : 'bad'
  const gwTitle = !g ? 'LiteLLM gateway: checking' : gwUp ? 'LiteLLM gateway: answering' : `LiteLLM gateway: ${g.auth === 'rejected' ? 'rejects the key' : g.problem ?? 'not answering'}`
  const alerts = alertsFor(fleet.entries, latest, gateway.data ?? undefined, readonly)
  const [view, setView] = useView()

  return (
    <>
      <header>
        <a className="brand" href="#/"><b>DGX</b>-kit{latest?.gpu?.name && <span className="brand-gpu" title="GPU">{latest.gpu.name}</span>}{version && <span className="ver" title="DGX-kit version">{/^\d/.test(version) ? `v${version}` : version}</span>}</a>
        {route.page === 'home' && latest && <Health alerts={alerts} entries={fleet.entries} />}
        <span className="grow" />
        {route.page !== 'home' && <a className="navlink" href="#/">Back to dashboard</a>}
        <a className={`navlink ${route.page === 'library' ? 'on' : ''}`} href={href({ page: 'library' })}>Models on disk</a>
        <a className={`navlink ${route.page === 'settings' ? 'on' : ''}`} href={href({ page: 'settings' })}>Settings</a>
        {readonly && <span className="pill warn" title="DGX-kit won't start, stop, pull or download anything on this box">Read-only</span>}
        <a className="conn" href={href({ page: 'settings', arg: 'gateway' })} title={gwTitle}><span className={`dot ${gwState}`} />LiteLLM</a>
        <span className="conn" title={connected ? 'Receiving live data' : 'Reconnecting'}><span className={`dot ${connected ? 'on' : ''}`} />{connected ? 'Live' : 'Reconnecting'}</span>
        {canSignOut && <button className="ghost" onClick={onSignOut}>Sign out</button>}
      </header>
      <main>
        {route.page === 'settings' ? <Settings canChangePassword={canSignOut} tab={route.arg} />
          : route.page === 'library' ? <LibraryPage />
          : !latest || !layout ? <p className="muted">Waiting for the first reading…</p>
          : <Home view={view} setView={setView} latest={latest} history={history} entries={fleet.entries} orphans={fleet.orphans} alerts={alerts} loaded={fleet.loaded}
              readonly={readonly} layout={layout} update={update} slo={slo} setSlo={setSlo} sel={route.arg} onChanged={fleet.reload} />}
      </main>
    </>
  )
}

/** Shown only when something needs you; each problem names its model and opens it. */
function Health({ alerts, entries }: { alerts: Alert[]; entries: Entry[] }) {
  const [open, setOpen] = useState(false)
  const serving = entries.filter((e) => e.state === 'running').length
  if (!alerts.length) return null
  const level = alerts.some((a) => a.level === 'bad') ? 'bad' : alerts.length ? 'warn' : 'ok'
  const first = alerts[0]
  const pick = (a: Alert) => { setOpen(false); if (a.model) select(a.model) }
  return (
    <div className={`health ${level}`} role="status">
      <span className={`dot ${level}`} aria-hidden />
      {first ? (
        <>
          <button className="link" onClick={() => pick(first)}><b>{first.short}</b></button>
          {alerts.length > 1 && <button className="link muted" aria-expanded={open} onClick={() => setOpen(!open)}>and {alerts.length - 1} more</button>}
          <span className="muted">· {serving} serving</span>
        </>
      ) : null}
      {open && (
        <ul className="health-list">
          {alerts.map((a, i) => (
            <li key={i} className={a.level}>
              <span aria-hidden>{a.level === 'bad' ? '●' : '▲'}</span>
              <span className="grow">{a.text}</span>
              {a.model && <button onClick={() => pick(a)}>Open</button>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** Map or list is a per-browser choice: each screen keeps its own. */
function useView(): ['map' | 'list', (v: 'map' | 'list') => void] {
  const read = () => { try { return localStorage.getItem('dgxkit.view') === 'list' ? 'list' : 'map' } catch { return 'map' } }
  const [view, set] = useState<'map' | 'list'>(read)
  return [view, (v) => { set(v); try { localStorage.setItem('dgxkit.view', v) } catch { /* private window */ } }]
}
