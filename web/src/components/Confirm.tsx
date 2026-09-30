import { createContext, useCallback, useContext, useEffect, useId, useRef, useState, type ReactNode } from 'react'

/** A modal on the native <dialog>: it traps focus, closes on Escape, and dims the page behind it. */
export function Modal({ open, onClose, title, children }: { open: boolean; onClose: () => void; title: string; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null)
  const id = useId()
  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])
  return (
    <dialog ref={ref} className="modal" aria-labelledby={id} onClose={onClose}
      onClick={(e) => { if (e.target === ref.current) onClose() }}>
      {open && <><h3 id={id}>{title}</h3>{children}</>}
    </dialog>
  )
}

export type Ask = { title: string; body?: ReactNode; label?: string; danger?: boolean }
const Ctx = createContext<(a: Ask) => Promise<boolean>>(() => Promise.resolve(false))

/** confirm({ title, body, label, danger }) opens a modal and resolves true only on the confirm button. */
export const useConfirm = () => useContext(Ctx)

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [ask, setAsk] = useState<(Ask & { resolve: (v: boolean) => void }) | null>(null)
  const confirm = useCallback((a: Ask) => new Promise<boolean>((resolve) => setAsk({ ...a, resolve })), [])
  const done = (v: boolean) => { ask?.resolve(v); setAsk(null) }
  return (
    <Ctx.Provider value={confirm}>
      {children}
      <Modal open={!!ask} onClose={() => done(false)} title={ask?.title ?? ''}>
        {ask?.body && <div className="modal-body">{ask.body}</div>}
        <div className="row modal-actions">
          <button autoFocus onClick={() => done(false)}>Cancel</button>
          <button className={ask?.danger ? 'danger' : 'primary'} onClick={() => done(true)}>{ask?.label ?? 'OK'}</button>
        </div>
      </Modal>
    </Ctx.Provider>
  )
}
