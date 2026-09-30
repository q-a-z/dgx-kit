import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { api } from '../api'

export type TileConf = { id: string; w: number; h?: number; hidden?: boolean; opts?: Record<string, number> }
export type Layout = {
  model: TileConf[]; hardware: TileConf[]; panel: TileConf[]
  model_name?: string; stat?: string; hw_open?: boolean
  /** Home picture: memory map (C) or model list (A). */
  view?: 'map' | 'list'
  /** The memory map drawn as the GPU process table instead. */
  map_table?: boolean
}
export type Group = 'model' | 'hardware'

/** Keep the saved order and settings, drop tiles that no longer exist, append new ones. */
export function merge(saved: TileConf[] | undefined, defaults: TileConf[]): TileConf[] {
  const known = new Map(defaults.map((d) => [d.id, d]))
  const kept = (saved ?? []).filter((t) => known.has(t.id)).map((t) => ({ ...known.get(t.id)!, ...t }))
  const seen = new Set(kept.map((t) => t.id))
  return [...kept, ...defaults.filter((d) => !seen.has(d.id))]
}

/** The dashboard layout, loaded from and saved to the service so every browser shares it. */
export function useLayout(defaults: Layout) {
  const [layout, setLayout] = useState<Layout | null>(null)
  const timer = useRef<number | undefined>(undefined)
  const defaultsRef = useRef(defaults)

  useEffect(() => {
    const d = defaultsRef.current
    api<{ layout: Partial<Layout> | null }>('/api/layout')
      .then(({ layout: saved }) => setLayout({ ...d, ...saved, model: merge(saved?.model, d.model), hardware: merge(saved?.hardware, d.hardware), panel: merge(saved?.panel, d.panel) }))
      .catch(() => setLayout(d))
  }, [])

  const save = useCallback((next: Layout | null) => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => api('/api/layout', { method: 'PUT', json: { layout: next } }).catch(() => {}), 400)
  }, [])

  const update = useCallback((patch: Partial<Layout>) => {
    setLayout((l) => {
      const next = { ...(l ?? defaultsRef.current), ...patch }
      save(next)
      return next
    })
  }, [save])

  const reset = useCallback(() => {
    setLayout((l) => ({ ...defaultsRef.current, model_name: l?.model_name, stat: l?.stat, view: l?.view }))
    save(null)
  }, [save])

  return { layout, update, reset }
}

export type Option = { key: string; label: string; default: number; choices?: [number, string][] }

export type TileDef<C> = {
  id: string
  title: string | ((ctx: C) => string)
  sub?: (ctx: C) => string | undefined
  w: number
  options?: Option[]
  render: (ctx: C, opts: Record<string, number>) => ReactNode
}

export function optsOf<C>(def: TileDef<C>, conf: TileConf) {
  return Object.fromEntries((def.options ?? []).map((o) => [o.key, conf.opts?.[o.key] ?? o.default])) as Record<string, number>
}
