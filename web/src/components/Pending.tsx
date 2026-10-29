/** What a section shows before its data arrives, or when it can't be read. */
export function Pending({ error, what }: { error: string | null; what: string }) {
  return <p className={error ? 'bad' : 'muted'}>{error ? `Couldn’t read ${what}: ${error}` : `Reading ${what}…`}</p>
}
