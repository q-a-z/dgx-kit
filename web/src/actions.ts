import { useState } from 'react'
import { api, ApiError } from './api'
import { useConfirm } from './components/Confirm'
import type { Entry } from './fleet'

export type Msg = { text: string; bad?: boolean }

/** Start, stop, restart, download and cancel for one model, with the reason when a control is off. */
export function useModelActions(entry: Entry, readonly: boolean, onChanged: () => void) {
  const [busy, setBusy] = useState<string | null>(null)
  const [msg, setMsg] = useState<Msg | null>(null)
  const q = encodeURIComponent(entry.name)
  const ask = useConfirm()

  const call = async (what: string, path: string, method = 'POST', retry = true) => {
    setBusy(what)
    setMsg(null)
    try {
      await api(path, { method })
      return true
    } catch (e) {
      if (retry && e instanceof ApiError && await ask({ title: 'Do it anyway?', body: e.message, label: 'Do it anyway', danger: true })) {
        const ok = await api(`${path}${path.includes('?') ? '&' : '?'}force=1`, { method }).then(() => true, (x: Error) => { setMsg({ text: x.message, bad: true }); return false })
        return ok
      }
      setMsg({ text: `${entry.name}: ${(e as Error).message}`, bad: true })
      return false
    } finally {
      setBusy(null)
      onChanged()
    }
  }
  const doStop = () => call('Stopping', `/api/models/${q}/stop`)
  const stop = async () => (await ask({ title: `Stop ${entry.name}?`, body: 'It stops serving right away. Requests in progress are lost, and it has to load again to come back.', label: 'Stop', danger: true })) && doStop()
  const start = () => call('Starting', `/api/models/${q}/start`, 'POST', false)
  const restart = async () => (await ask({ title: `Restart ${entry.name}?`, body: 'It stops, then loads again; that takes minutes and requests in progress are lost.', label: 'Restart', danger: true })) && (await doStop()) && start()
  const download = () => call('Starting download', `/api/models/${q}/download`, 'POST', false)
  const cancel = () => {
    const repo = entry.download?.repo
    return repo ? call('Cancelling', `/api/downloads/${repo}`, 'DELETE', false) : Promise.resolve(false)
  }
  const pause = () => {
    const repo = entry.download?.repo
    return repo ? call('Pausing', `/api/downloads/${repo}/pause`, 'POST', false) : Promise.resolve(false)
  }
  const resume = () => {
    const repo = entry.download?.repo
    return repo ? call('Resuming', `/api/downloads/${repo}/resume`, 'POST', false) : Promise.resolve(false)
  }
  const remove = async () => {
    const ok = await ask({ title: `Remove ${entry.name} from DGX-kit?`, danger: true, label: 'Remove',
      body: 'Only the model’s settings and history here are removed. Its files on disk stay where they are; deleting files is in Settings.' })
    return ok && call('Removing', `/api/models/${q}`, 'DELETE', false)
  }
  const why = !entry.managed ? 'Started outside DGX-kit: watch only.'
    : readonly ? 'Read-only: DGX-kit won’t start, stop or download anything on this box.' : null
  return { busy, msg, setMsg, start, stop, restart, download, cancel, pause, resume, remove, locked: readonly || !entry.managed || !!busy, why }
}
