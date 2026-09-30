import { useCallback, useEffect, useState } from 'react'
import { api } from './api'

/** GET a path every few seconds; `reload` refetches right away after an action. */
export function usePoll<T>(path: string, everyMs = 3000) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const reload = useCallback(() => {
    api<T>(path)
      .then((d) => {
        setData(d)
        setError(null)
      })
      .catch((e: Error) => setError(e.message))
  }, [path])
  useEffect(() => {
    reload()
    const id = setInterval(reload, everyMs)
    return () => clearInterval(id)
  }, [reload, everyMs])
  return { data, error, reload }
}
