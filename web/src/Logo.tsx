/** The DGX-kit mark: a grid of cores with a lit signal path. The same drawing as public/favicon.svg. */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg className="logo" width={size} height={size} viewBox="0 0 64 64" aria-hidden>
      <rect width="64" height="64" rx="12" fill="#1d1e21" />
      <g fill="#4a4232"><circle cx="16" cy="32" r="4.5" /><circle cx="16" cy="48" r="4.5" /><circle cx="48" cy="32" r="4.5" /><circle cx="32" cy="48" r="4.5" /></g>
      <path d="M16 16L32 32L48 16M32 32V48" fill="none" stroke="#c9a86a" strokeWidth="2.5" />
      <g fill="#c9a86a"><circle cx="16" cy="16" r="4.5" /><circle cx="32" cy="32" r="4.5" /><circle cx="48" cy="16" r="4.5" /><circle cx="48" cy="48" r="4.5" /></g>
    </svg>
  )
}
