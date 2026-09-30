import { useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'
import { optsOf, type TileConf, type TileDef } from './layout'
export type { Option, TileDef } from './layout'

const COLS = 12
const MIN_H = 110
const SNAP_H = 10

/**
 * A 12-column grid of tiles. Any tile can be resized from its lower-right corner
 * (width snaps to columns, height to 10 px). Arranging adds moving, hiding and settings.
 */
export function TileGrid<C>({ defs, tiles, ctx, editing, onChange }: {
  defs: TileDef<C>[]
  tiles: TileConf[]
  ctx: C
  editing: boolean
  onChange: (tiles: TileConf[]) => void
}) {
  const grid = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState<string | null>(null)
  const [over, setOver] = useState<string | null>(null)
  const [settings, setSettings] = useState<string | null>(null)
  const [sizing, setSizing] = useState<{ id: string; w: number; h: number } | null>(null)
  const byId = new Map(defs.map((d) => [d.id, d]))
  const visible = tiles.filter((t) => !t.hidden && byId.has(t.id))
  const hidden = tiles.filter((t) => t.hidden && byId.has(t.id))

  const patch = (id: string, p: Partial<TileConf>) => onChange(tiles.map((t) => (t.id === id ? { ...t, ...p } : t)))
  const move = (id: string, to: string | number) => {
    const from = tiles.findIndex((t) => t.id === id)
    const rest = tiles.filter((t) => t.id !== id)
    const at = typeof to === 'number' ? Math.max(0, Math.min(rest.length, from + to)) : rest.findIndex((t) => t.id === to)
    rest.splice(at < 0 ? rest.length : at, 0, tiles[from])
    onChange(rest)
  }
  const title = (d: TileDef<C>) => (typeof d.title === 'function' ? d.title(ctx) : d.title)

  const startResize = (e: ReactPointerEvent, t: TileConf) => {
    e.preventDefault()
    e.stopPropagation()
    const tile = (e.currentTarget as HTMLElement).closest('.tile') as HTMLElement
    const g = grid.current
    if (!tile || !g) return
    const gap = parseFloat(getComputedStyle(g).columnGap) || 0
    const col = (g.clientWidth - gap * (COLS - 1)) / COLS
    const x0 = e.clientX
    const y0 = e.clientY
    const w0 = tile.offsetWidth
    const h0 = tile.offsetHeight
    let next = { id: t.id, w: t.w, h: h0 }
    const onMove = (ev: PointerEvent) => {
      const w = Math.max(2, Math.min(COLS, Math.round((w0 + ev.clientX - x0 + gap) / (col + gap))))
      const h = Math.max(MIN_H, Math.round((h0 + ev.clientY - y0) / SNAP_H) * SNAP_H)
      next = { id: t.id, w, h }
      setSizing(next)
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      setSizing(null)
      patch(t.id, { w: next.w, h: next.h === h0 ? t.h : next.h })
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  return (
    <div ref={grid} className={`tiles ${editing ? 'editing' : ''}`}>
      {visible.map((t) => {
        const d = byId.get(t.id)!
        const opts = optsOf(d, t)
        const live = sizing?.id === t.id ? sizing : null
        const w = live?.w ?? t.w
        const h = live?.h ?? t.h
        return (
          <section
            key={t.id}
            className={`card tile w${w} ${over === t.id && dragging !== t.id ? 'drop' : ''} ${dragging === t.id ? 'lifted' : ''} ${live ? 'sizing' : ''}`}
            style={{ '--w': w, height: h ? `${h}px` : undefined } as CSSProperties}
            draggable={editing}
            onDragStart={(e) => { setDragging(t.id); e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', t.id) }}
            onDragEnd={() => { setDragging(null); setOver(null) }}
            onDragOver={(e) => { if (dragging) { e.preventDefault(); setOver(t.id) } }}
            onDrop={(e) => { e.preventDefault(); if (dragging && dragging !== t.id) move(dragging, t.id); setDragging(null); setOver(null) }}
          >
            <div className="tile-head">
              {editing && <span className="grip" aria-hidden>⠿</span>}
              <h2>{title(d)}{d.sub?.(ctx) && <span>{d.sub(ctx)}</span>}</h2>
              {editing && (
                <span className="tile-tools">
                  <button title="Move earlier" aria-label="Move earlier" onClick={() => move(t.id, -1)}>←</button>
                  <button title="Move later" aria-label="Move later" onClick={() => move(t.id, 1)}>→</button>
                  {d.options && <button title="Settings" aria-label="Settings" aria-pressed={settings === t.id} onClick={() => setSettings(settings === t.id ? null : t.id)}>⚙</button>}
                  <button title="Hide" aria-label="Hide" onClick={() => patch(t.id, { hidden: true })}>✕</button>
                </span>
              )}
            </div>
            {editing && settings === t.id && d.options && (
              <div className="tile-opts">
                {d.options.map((o) => (
                  <label key={o.key}>
                    {o.label}
                    {o.choices ? (
                      <select value={opts[o.key]} onChange={(e) => patch(t.id, { opts: { ...t.opts, [o.key]: Number(e.target.value) } })}>
                        {o.choices.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                      </select>
                    ) : (
                      <input type="number" value={opts[o.key]} onChange={(e) => e.target.value !== '' && patch(t.id, { opts: { ...t.opts, [o.key]: Number(e.target.value) } })} />
                    )}
                  </label>
                ))}
                <button onClick={() => patch(t.id, { opts: undefined })}>Defaults</button>
              </div>
            )}
            <div className="tile-body">{d.render(ctx, opts)}</div>
            <div className="resize" role="separator" aria-label={`Resize ${title(d)}`} title="Drag to resize (double-click to reset height)"
              onPointerDown={(e) => startResize(e, t)} onDoubleClick={() => patch(t.id, { h: undefined })} />
            {live && <span className="size-badge">{w} of 12 columns{h ? `, ${h} px tall` : ''}</span>}
          </section>
        )
      })}
      {editing && hidden.length > 0 && (
        <section className="card tile add">
          <span className="muted">Hidden:</span>
          {hidden.map((t) => <button key={t.id} onClick={() => patch(t.id, { hidden: false })}>Show {title(byId.get(t.id)!)}</button>)}
        </section>
      )}
    </div>
  )
}
