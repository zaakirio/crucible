import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, fmtDate, type RunSummary } from '../api'
import { Empty, ErrorNote, LabelBar, LineagePill, PassBar, SkeletonRows } from '../components'

export default function Runs() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState('')
  const [picked, setPicked] = useState<number[]>([])
  const nav = useNavigate()

  useEffect(() => {
    api.runs().then(setRuns).catch(e => setError(String(e.message ?? e)))
  }, [])

  function togglePick(id: number) {
    setPicked(p => p.includes(id) ? p.filter(x => x !== id) : [...p.slice(-1), id])
  }

  if (error) return <ErrorNote error={error} />

  return (
    <div className="fade-in">
      <div className="page-head">
        <h1>Runs</h1>
        <div className="spacer" />
        <div className="bar-legend" aria-hidden="true">
          <span><span className="dot" style={{ background: 'var(--complied)' }} />complied</span>
          <span><span className="dot" style={{ background: 'var(--hedged)' }} />hedged</span>
          <span><span className="dot" style={{ background: 'var(--refused)' }} />refused</span>
        </div>
        <button
          className="btn primary"
          disabled={picked.length !== 2}
          onClick={() => nav(`/compare?a=${picked[0]}&b=${picked[1]}`)}
          title={picked.length !== 2 ? 'Select two runs with the checkboxes to diff them' : undefined}
        >
          Diff {picked.length === 2 ? `#${picked[0]} vs #${picked[1]}` : 'runs'}
        </button>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th aria-label="select for diff" />
              <th className="r">#</th>
              <th>Model</th>
              <th>Started</th>
              <th className="r">Results</th>
              <th>Capability</th>
              <th>Refusal profile</th>
              <th className="r">ppl</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {runs === null && <SkeletonRows cols={9} />}
            {runs?.map(r => (
              <tr
                key={r.id}
                className={`rowlink${picked.includes(r.id) ? ' selected' : ''}`}
                tabIndex={0}
                onClick={() => nav(`/runs/${r.id}`)}
                onKeyDown={e => { if (e.key === 'Enter') nav(`/runs/${r.id}`) }}
              >
                <td onClick={e => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    aria-label={`select run ${r.id} for diff`}
                    checked={picked.includes(r.id)}
                    onChange={() => togglePick(r.id)}
                  />
                </td>
                <td className="r num">{r.id}</td>
                <td>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 500 }}>{r.model_name ?? r.model_file}</span>
                    {r.quant && <span className="pill neutral">{r.quant}</span>}
                    <LineagePill lineage={r.lineage} />
                  </div>
                </td>
                <td className="num" style={{ color: 'var(--ink-mid)', whiteSpace: 'nowrap' }}>{fmtDate(r.started_at)}</td>
                <td className="r num">{r.n_results}</td>
                <td style={{ minWidth: 130 }}>
                  {r.n_graded > 0 ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <PassBar passed={r.n_passed} graded={r.n_graded} />
                      <span className="num" style={{ fontSize: '0.8rem', color: 'var(--ink-mid)' }}>
                        {r.n_passed}/{r.n_graded}
                      </span>
                    </div>
                  ) : <span style={{ color: 'var(--ink-muted)' }}>-</span>}
                </td>
                <td style={{ minWidth: 130 }}>
                  <LabelBar complied={r.n_complied} hedged={r.n_hedged} refused={r.n_refused} />
                </td>
                <td className="r num">{r.ppl ? r.ppl.toFixed(2) : '-'}</td>
                <td>
                  <span className={`pill ${r.status === 'done' ? 'neutral' : 'hedged'}`}>{r.status}</span>
                </td>
              </tr>
            ))}
            {runs?.length === 0 && (
              <tr><td colSpan={9}>
                <Empty title="No runs yet">
                  Run an eval first: <code>crucible eval --server http://localhost:11434/v1 --model-name my-model --judge claude</code>, then refresh.
                </Empty>
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
