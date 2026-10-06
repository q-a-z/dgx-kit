/** List prices in US dollars per million tokens, as published on this date. Check the vendors' pages before relying on them. */
/** Shares of input tokens assumed to be cache reads, one extra column each. */
export const CACHE_SHARES = [0.5, 0.9]
export const PRICES_AS_OF = '2026-10-06'

export type Price = { vendor: 'Claude' | 'OpenAI'; model: string; in: number; cached: number; out: number }

export const PRICES: Price[] = [
  { vendor: 'Claude', model: 'Fable 5.1', in: 10, cached: 0.25, out: 50 },
  { vendor: 'Claude', model: 'Opus 5.5', in: 4, cached: 0.2, out: 20 },
  { vendor: 'Claude', model: 'Sonnet 5.5', in: 2, cached: 0.2, out: 10 },
  { vendor: 'Claude', model: 'Haiku 4.5', in: 1, cached: 0.1, out: 5 },
  { vendor: 'OpenAI', model: 'GPT-5.5', in: 5, cached: 0.5, out: 30 },
  { vendor: 'OpenAI', model: 'GPT-5', in: 1.25, cached: 0.125, out: 10 },
  { vendor: 'OpenAI', model: 'GPT-5 nano', in: 0.05, cached: 0.005, out: 0.4 },
]
