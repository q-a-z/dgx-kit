import { useEffect, useState } from 'react'

/**
 * Three pages: home, the models and drafts on disk, and settings. On home the hash also names what the side panel shows:
 * #/model/<name> for a model, #/run to add one. Picking a model never leaves the page.
 */
export type Route = { page: 'home' | 'settings' | 'library'; arg?: string }
export const RUN = '+run'

export function parse(hash: string): Route {
  const [page, ...rest] = decodeURIComponent(hash.replace(/^#\/?/, '')).split('/')
  if (page === 'model' && rest.length) return { page: 'home', arg: rest.join('/') }
  if (page === 'run') return { page: 'home', arg: RUN }
  if (page === 'settings') return { page, arg: rest.join('/') || undefined }  // #/settings/<tab>
  if (page === 'library') return { page }
  return { page: 'home' }
}

export const href = (r: Route) =>
  r.page === 'settings' ? `#/settings${r.arg ? `/${encodeURIComponent(r.arg)}` : ''}` : r.page === 'library' ? '#/library' : r.arg === RUN ? '#/run' : r.arg ? `#/model/${encodeURIComponent(r.arg)}` : '#/'
export const go = (r: Route) => { location.hash = href(r) }
export const select = (name: string | undefined) => go({ page: 'home', arg: name })

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(location.hash))
  useEffect(() => {
    const on = () => setRoute((was) => {
      const next = parse(location.hash)
      if (next.page !== was.page) window.scrollTo(0, 0)
      return next
    })
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}
