import { useEffect, useState } from 'react'
import type { Snapshot } from './api'

const KEEP = 900 // 15 minutes at one sample per second

/** Latest snapshot plus rolling history from the service's server-sent event stream. */
export function useStream() {
  const [history, setHistory] = useState<Snapshot[]>([])
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetch('/api/history')
      .then((r) => r.json())
      .then((h: Snapshot[]) => {
        // Keep anything the stream delivered while history was loading.
        if (!cancelled) setHistory((live) => [...h.filter((s) => !live.length || s.t < live[0].t), ...live].slice(-KEEP))
      })
      .catch(() => {})

    const es = new EventSource('/api/stream')
    es.onopen = () => setConnected(true)
    es.onerror = () => setConnected(false) // EventSource reconnects on its own
    es.onmessage = (e) => {
      const snap = JSON.parse(e.data) as Snapshot
      setHistory((h) => (h.length >= KEEP ? [...h.slice(1 - KEEP), snap] : [...h, snap]))
    }
    return () => {
      cancelled = true
      es.close()
    }
  }, [])

  return { latest: history.at(-1) ?? null, history, connected }
}
