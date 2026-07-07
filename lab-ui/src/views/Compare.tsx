import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, fmtDate, shortHash, type ComparePayload, type RunSummary } from '../api'
import { Chip, Empty, ErrorNote, LineagePill } from '../components'

export default function Compare() {
  const [params, setParams] = useSearchParams()
  const a = params.get('a') ? Number(params.get('a')) : null
  const b = params.get('b') ? Number(params.get('b')) : null
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [payload, setPayload] = useState<ComparePayload | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.runs().then(setRuns).catch(e => setError(String(e.message ?? e)))
  }, [])

  useEffect(() => {
    let stale = false
    setPayload(null)
    setError('')
    if (a !== null && b !== null) {
      api.compare(a, b)
        .then(p => { if (!stale) setPayload(p) })
        .catch(e => { if (!stale) setError(String(e.message ?? e)) })
    }
    return () => { stale = true }
  }, [a, b])

  function setSide(side: 'a' | 'b', id: string) {
    setParams(prev => {
      const p = new URLSearchParams(prev)
      if (id) p.set(side, id); else p.delete(side)
      return p
    })
  }

  const runLabel = (r: RunSummary) =>
    `#${r.id} ${r.model_name ?? r.model_file} ${r.quant ?? ''} (${fmtDate(r.started_at)})`

  return (
    <div className="fade-in">
      <div className="page-head">
        <h1>Diff runs</h1>
        <div className="spacer" />
        <div className="controls">
          <select className="select" value={a ?? ''} onChange={e => setSide('a', e.target.value)} aria-label="baseline run">
            <option value="">baseline…</option>
            {runs.map(r => <option key={r.id} value={r.id}>{runLabel(r)}</option>)}
          </select>
          <span style={{ color: 'var(--ink-muted)' }}>vs</span>
          <select className="select" value={b ?? ''} onChange={e => setSide('b', e.target.value)} aria-label="candidate run">
            <option value="">candidate…</option>
            {runs.map(r => <option key={r.id} value={r.id}>{runLabel(r)}</option>)}
          </select>
        </div>
      </div>

      {error && <ErrorNote error={error} />}

      {a === null || b === null ? (
        <Empty title="Pick two runs">
          Baseline on the left, candidate on the right - the classic use is base model vs abliterated, or Q8 vs Q4.
        </Empty>
      ) : payload && (
        <>
          <div className="compare-heads">
            <RunHead title="A · baseline" run={payload.a} />
            <RunHead title="B · candidate" run={payload.b} />
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Category</th>
                  <th className="r">A</th>
                  <th className="r">B</th>
                  <th className="r">delta (B - A)</th>
                </tr>
              </thead>
              <tbody>
                {payload.rows.map(row => (
                  <tr key={row.category} className={row.flagged ? 'flagged' : ''}>
                    <td className="mono">
                      {row.category}
                      {row.flagged && <span className="pill fail" style={{ marginLeft: 8 }}>capability drop ≥15pp</span>}
                    </td>
                    <td className="r num">{row.value_a}</td>
                    <td className="r num">{row.value_b}</td>
                    <td className={`r num ${deltaClass(row.delta)}`}>{row.delta || '·'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p style={{ color: 'var(--ink-muted)', marginTop: 12, maxWidth: '75ch' }}>
            Refusal categories read as complied/refused counts ("12c/8r"), and their delta is complied-count movement:
            positive means the candidate answers more than the baseline.
            Capability rows are pass rates; flagged rows dropped 15 percentage points or more.
          </p>
        </>
      )}
    </div>
  )
}

function deltaClass(delta: string): string {
  if (!delta) return 'delta-zero'
  if (delta.startsWith('+0') || delta === '+0%') return 'delta-zero'
  return delta.startsWith('-') ? 'delta-neg' : 'delta-pos'
}

function RunHead({ title, run }: { title: string; run: RunSummary }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
        <div className="spacer" style={{ flex: 1 }} />
        <LineagePill lineage={run.lineage} />
      </div>
      <div className="panel-body">
        <p style={{ margin: '0 0 8px', fontWeight: 500 }}>
          #{run.id} · {run.model_name ?? run.model_file} {run.quant && <span className="pill neutral">{run.quant}</span>}
        </p>
        <div className="provenance" style={{ margin: 0 }}>
          <Chip label="started">{fmtDate(run.started_at)}</Chip>
          <Chip label="model"><span className="mono">{shortHash(run.model_sha256)}</span></Chip>
          {run.ppl != null && <Chip label="ppl"><span className="num">{run.ppl.toFixed(2)}</span></Chip>}
        </div>
      </div>
    </div>
  )
}
