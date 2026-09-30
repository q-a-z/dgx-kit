/** A small stroke icon set, so every action reads the same wherever it sits. */
const PATHS: Record<string, string> = {
  restart: 'M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2.5v3h-3',
  stop: 'M4.5 4.5h7v7h-7z',
  pause: 'M5.5 4v8M10.5 4v8',
  start: 'M5 3.5l8 4.5-8 4.5z',
  logs: 'M3 4h10M3 8h10M3 12h6',
  settings: 'M3 5h6M12 5h1M3 11h1M7 11h6M10.5 3.5v3M5.5 9.5v3',
  download: 'M8 2.5v8M4.5 7.5L8 11l3.5-3.5M3 13.5h10',
  cancel: 'M4.5 4.5l7 7M11.5 4.5l-7 7',
  plus: 'M8 3v10M3 8h10',
  copy: 'M5.5 5.5h7v7h-7zM3.5 10.5v-7h7',
  more: 'M3.5 8h.01M8 8h.01M12.5 8h.01',
  close: 'M4.5 4.5l7 7M11.5 4.5l-7 7',
  cpu: 'M4.5 4.5h7v7h-7zM6.5 2v2.5M9.5 2v2.5M6.5 11.5V14M9.5 11.5V14M2 6.5h2.5M2 9.5h2.5M11.5 6.5H14M11.5 9.5H14',
  gpu: 'M1.5 4.5h13v7h-13zM5.5 8a2 2 0 1 0 4 0a2 2 0 1 0-4 0M12 6.5v3M4 11.5v2M7 11.5v2',
}

export function Icon({ name, size = 16 }: { name: keyof typeof PATHS | string; size?: number }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor"
      strokeWidth={name === 'more' ? 2.6 : 1.6} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={PATHS[name] ?? ''} />
    </svg>
  )
}
