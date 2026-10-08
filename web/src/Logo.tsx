/** The DGX-kit mark: a grid of cores with a lit signal path. The same drawing as public/favicon.svg. */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg className="logo" width={size} height={size} viewBox="0 0 64 64" aria-hidden>
      <rect width="64" height="64" rx="12" fill="#0b0b0b" />
      <g fill="#2C3D10"><circle cx="16" cy="32" r="4.5" /><circle cx="16" cy="48" r="4.5" /><circle cx="48" cy="32" r="4.5" /><circle cx="32" cy="48" r="4.5" /></g>
      <path d="M16 16L32 32L48 16M32 32V48" fill="none" stroke="#76B900" strokeWidth="2.5" />
      <g fill="#76B900"><circle cx="16" cy="16" r="4.5" /><circle cx="32" cy="32" r="4.5" /><circle cx="48" cy="16" r="4.5" /><circle cx="48" cy="48" r="4.5" /></g>
    </svg>
  )
}
