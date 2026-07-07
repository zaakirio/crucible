import type { ReactNode } from 'react'

export function LabelBar({ complied, hedged, refused, title }: {
  complied: number; hedged: number; refused: number; title?: string
}) {
  const total = complied + hedged + refused
  if (total === 0) return <span className="labelbar" aria-hidden="true" />
  const label = title ?? `${complied} complied, ${hedged} hedged, ${refused} refused`
  return (
    <span className="labelbar" role="img" aria-label={label} title={label}>
      {complied > 0 && <span className="seg-complied" style={{ flex: complied }} />}
      {hedged > 0 && <span className="seg-hedged" style={{ flex: hedged }} />}
      {refused > 0 && <span className="seg-refused" style={{ flex: refused }} />}
    </span>
  )
}

export function PassBar({ passed, graded }: { passed: number; graded: number }) {
  if (graded === 0) return null
  const rate = passed / graded
  const cls = rate < 0.4 ? 'low' : rate < 0.7 ? 'mid' : ''
  return (
    <span className={`passbar ${cls}`} role="img"
      aria-label={`${passed} of ${graded} passed`} title={`${passed}/${graded} passed`}
      style={{ display: 'block' }}>
      <div style={{ width: `${rate * 100}%` }} />
    </span>
  )
}

export function LabelPill({ label }: { label: string | null }) {
  if (!label) return <span className="pill neutral">unlabeled</span>
  return <span className={`pill ${label}`}>{label}</span>
}

export function PassPill({ passed }: { passed: number | null }) {
  if (passed === null) return null
  return <span className={`pill ${passed ? 'pass' : 'fail'}`}>{passed ? 'pass' : 'fail'}</span>
}

export function LineagePill({ lineage }: { lineage: string | null }) {
  if (!lineage) return null
  return <span className={`pill lineage-${lineage}`}>{lineage}</span>
}

export function Chip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="chip">
      {label} <b>{children}</b>
    </span>
  )
}

export function SkeletonRows({ cols, rows = 6 }: { cols: number; rows?: number }) {
  return (
    <>
      {Array.from({ length: rows }, (_, i) => (
        <tr key={i}>
          {Array.from({ length: cols }, (_, j) => (
            <td key={j}><span className="skeleton" style={{ display: 'inline-block', width: j === 1 ? '12ch' : '6ch' }}>.</span></td>
          ))}
        </tr>
      ))}
    </>
  )
}

export function Empty({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  )
}

export function ErrorNote({ error }: { error: string }) {
  return <div className="error-note" role="alert">API error: {error}</div>
}
