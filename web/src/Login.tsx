import { useState } from 'react'
import { api } from './api'
import { toast } from './toast'

export function Login({ onDone, version }: { onDone: () => void; version?: string }) {
  const [pw, setPw] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await api('/api/login', { method: 'POST', json: { password: pw } })
      onDone()
    } catch (e) {
      setErr((e as Error).message)
    }
  }
  return (
    <form className="card login" onSubmit={submit}>
      <h2>DGX-kit{version && <span className="ver">{/^\d/.test(version) ? `v${version}` : version}</span>}</h2>
      <label>Admin password<input type="password" autoFocus value={pw} onChange={(e) => setPw(e.target.value)} /></label>
      {err && <p className="bad">{err}</p>}
      <button className="primary" type="submit" disabled={!pw}>Sign in</button>
    </form>
  )
}

export function ChangePassword() {
  const [old, setOld] = useState('')
  const [next, setNext] = useState('')
  const [msg, setMsg] = useState<string | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await api('/api/password', { method: 'POST', json: { old, new: next } })
      setMsg('Password changed. Other signed-in browsers were signed out.')
      toast('Password changed.')
      setOld('')
      setNext('')
    } catch (e) {
      setMsg((e as Error).message)
    }
  }
  return (
    <form className="card wide" onSubmit={submit}>
      <h2>Admin password</h2>
      <div className="row">
        <label>Current<input type="password" value={old} onChange={(e) => setOld(e.target.value)} /></label>
        <label>New (8+ characters)<input type="password" value={next} onChange={(e) => setNext(e.target.value)} /></label>
        <button type="submit" disabled={next.length < 8}>Change password</button>
      </div>
      {msg && <p className="muted">{msg}</p>}
    </form>
  )
}
