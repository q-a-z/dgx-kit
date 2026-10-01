import { useEffect, useState } from 'react'

type Toast = { id: number; text: string; bad: boolean }
const listeners = new Set<(t: Toast) => void>()
let counter = 0

/** A short message in the corner of the screen, for "Saved" and the like. */
export const toast = (text: string, bad = false) => { const t = { id: ++counter, text, bad }; listeners.forEach((l) => l(t)) }

export function Toasts() {
  const [items, setItems] = useState<Toast[]>([])
  useEffect(() => {
    const on = (t: Toast) => {
      setItems((xs) => [...xs, t])
      window.setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== t.id)), t.bad ? 6000 : 3500)
    }
    listeners.add(on)
    return () => { listeners.delete(on) }
  }, [])
  return (
    <div className="toasts" role="status" aria-live="polite">
      {items.map((t) => <div key={t.id} className={`toast ${t.bad ? 'bad' : 'ok'}`}>{t.bad ? '▲ ' : '✓ '}{t.text}</div>)}
    </div>
  )
}
