import { useEffect, useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { api, fmtBytes, fmtDate, shortHash, type ResultRow, type RunDetail as RunDetailT } from '../api'
import {
  Chip, Empty, ErrorNote, LabelBar, LabelPill, LineagePill, PassBar, PassPill,
} from '../components'

function isLabelCategory(c: { n_graded: number; n_complied: number; n_hedged: number; n_refused: number }) {
  return c.n_graded === 0 && c.n_complied + c.n_hedged + c.n_refused > 0
}

export default function RunDetail() {
  const { id } = useParams()
  const runId = Number(id)
  const [run, setRun] = useState<RunDetailT | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let stale = false
    setRun(null)
    setError('')
    api.run(runId)
      .then(r => { if (!stale) setRun(r) })
      .catch(e => { if (!stale) setError(String(e.message ?? e)) })
    return () => { stale = true }
  }, [runId])

  if (error) return <ErrorNote error={error} />
  if (!run) return <p style={{ color: 'var(--ink-muted)' }}>Loading run {runId}…</p>

  const capability = run.categories.filter(c => !isLabelCategory(c))
  const refusal = run.categories.filter(isLabelCategory)

  return (
    <div className="fade-in">
      <div className="page-head">
        <h1>
          Run <span className="num">#{run.id}</span> · {run.model_name ?? run.model_file}
        </h1>
        {run.quant && <span className="pill neutral">{run.quant}</span>}
        <LineagePill lineage={run.lineage} />
        <div className="spacer" />
        <Link to={`/compare?a=${run.id}`} className="btn" style={{ textDecoration: 'none' }}>
          Diff against…
        </Link>
      </div>

      <div className="provenance">
        <Chip label="started">{fmtDate(run.started_at)}</Chip>
        <Chip label="model"><span className="mono">{shortHash(run.model_sha256)}</span></Chip>
        <Chip label="tests"><span className="mono">{shortHash(run.tests_sha256)}</span></Chip>
        {run.engine_tag && <Chip label="engine"><span className="mono">{run.engine_tag}</span></Chip>}
        {run.llama_cpp_commit && <Chip label="llama.cpp"><span className="mono">{run.llama_cpp_commit.slice(0, 9)}</span></Chip>}
        {run.ctx != null && <Chip label="ctx"><span className="num">{run.ctx}</span></Chip>}
        {run.ngl != null && <Chip label="ngl"><span className="num">{run.ngl}</span></Chip>}
        {run.model_size_bytes != null && <Chip label="size">{fmtBytes(run.model_size_bytes)}</Chip>}
        {run.ppl != null && <Chip label="ppl"><span className="num">{run.ppl.toFixed(2)}</span></Chip>}
        {run.server_url && <Chip label="server"><span className="mono">{run.server_url}</span></Chip>}
      </div>

      {run.flapping.length > 0 && (
        <p style={{ color: 'var(--warn)', margin: '0 0 14px' }}>
          {run.flapping.length} test{run.flapping.length > 1 ? 's' : ''} flapped across repetitions
          ({run.flapping.slice(0, 5).map(f => f.test_id).join(', ')}{run.flapping.length > 5 ? '…' : ''})
          - treat small deltas in those categories as noise.
        </p>
      )}

      <CategoryTables capability={capability} refusal={refusal} />

      <Inspector runId={runId} categories={run.categories.map(c => c.category)} />
    </div>
  )
}

function CategoryTables({ capability, refusal }: {
  capability: RunDetailT['categories']; refusal: RunDetailT['categories']
}) {
  const [, setParams] = useSearchParams()
  const pick = (cat: string) => setParams(prev => {
    const p = new URLSearchParams(prev)
    p.set('category', cat)
    return p
  })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 14 }}>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Capability</th><th className="r">passed</th><th style={{ width: '35%' }}>rate</th><th className="r">tok/s</th></tr>
          </thead>
          <tbody>
            {capability.map(c => (
              <tr key={c.category} className="rowlink" tabIndex={0}
                onClick={() => pick(c.category)}
                onKeyDown={e => { if (e.key === 'Enter') pick(c.category) }}>
                <td className="mono">{c.category}</td>
                <td className="r num">{c.n_passed}/{c.n_graded}</td>
                <td><PassBar passed={c.n_passed} graded={c.n_graded} /></td>
                <td className="r num">{c.avg_tps ? c.avg_tps.toFixed(0) : '-'}</td>
              </tr>
            ))}
            {capability.length === 0 && <tr><td colSpan={4} style={{ color: 'var(--ink-muted)' }}>none in this run</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Refusal profile</th><th className="r">c / h / r</th><th style={{ width: '35%' }}>profile</th></tr>
          </thead>
          <tbody>
            {refusal.map(c => (
              <tr key={c.category} className="rowlink" tabIndex={0}
                onClick={() => pick(c.category)}
                onKeyDown={e => { if (e.key === 'Enter') pick(c.category) }}>
                <td className="mono">{c.category}</td>
                <td className="r num">{c.n_complied} / {c.n_hedged} / {c.n_refused}</td>
                <td><LabelBar complied={c.n_complied} hedged={c.n_hedged} refused={c.n_refused} /></td>
              </tr>
            ))}
            {refusal.length === 0 && <tr><td colSpan={3} style={{ color: 'var(--ink-muted)' }}>none in this run</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Inspector({ runId, categories }: { runId: number; categories: string[] }) {
  const [params, setParams] = useSearchParams()
  const category = params.get('category') ?? ''
  const [label, setLabel] = useState('')
  const [status, setStatus] = useState('')
  const [q, setQ] = useState('')
  const [rows, setRows] = useState<ResultRow[] | null>(null)
  const [error, setError] = useState('')
  const [sel, setSel] = useState<number | null>(null)

  useEffect(() => {
    let stale = false
    setRows(null)
    setError('')
    api.results(runId, { category, label, status, q })
      .then(r => {
        if (stale) return
        setRows(r)
        setSel(r.length ? r[0].id : null)
      })
      .catch(e => { if (!stale) setError(String(e.message ?? e)) })
    return () => { stale = true }
  }, [runId, category, label, status, q])

  const selected = useMemo(() => rows?.find(r => r.id === sel) ?? null, [rows, sel])

  function setCategory(cat: string) {
    setParams(prev => {
      const p = new URLSearchParams(prev)
      if (cat) p.set('category', cat); else p.delete('category')
      return p
    })
  }

  function onListKey(e: React.KeyboardEvent) {
    if (!rows?.length || (e.key !== 'ArrowDown' && e.key !== 'ArrowUp')) return
    e.preventDefault()
    const idx = rows.findIndex(r => r.id === sel)
    const next = e.key === 'ArrowDown' ? Math.min(idx + 1, rows.length - 1) : Math.max(idx - 1, 0)
    setSel(rows[next].id)
  }

  return (
    <section className="section-gap">
      <div className="page-head">
        <h2>Transcripts</h2>
        <div className="spacer" />
        <div className="controls">
          <select className="select" value={category} onChange={e => setCategory(e.target.value)} aria-label="filter by category">
            <option value="">all categories</option>
            {categories.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <div className="seg-group" role="group" aria-label="filter by grade">
            {['', 'passed', 'failed'].map(s => (
              <button key={s || 'all'} className={status === s ? 'on' : ''} aria-pressed={status === s}
                onClick={() => { setStatus(s); setLabel('') }}>
                {s || 'all'}
              </button>
            ))}
          </div>
          <div className="seg-group" role="group" aria-label="filter by refusal label">
            {['complied', 'hedged', 'refused'].map(l => (
              <button key={l} className={label === l ? 'on' : ''} aria-pressed={label === l}
                onClick={() => { setLabel(label === l ? '' : l); setStatus('') }}>
                {l}
              </button>
            ))}
          </div>
          <input className="input" placeholder="search prompts and responses" value={q}
            aria-label="search prompts and responses"
            onChange={e => setQ(e.target.value)} style={{ width: 220 }} />
        </div>
      </div>

      {error && <ErrorNote error={error} />}

      <div className="inspector">
        <div className="panel result-list" onKeyDown={onListKey}>
          {rows === null && <p style={{ padding: 14, color: 'var(--ink-muted)' }}>Loading…</p>}
          {rows?.map(r => (
            <button key={r.id} className={`result-item${r.id === sel ? ' on' : ''}`}
              aria-pressed={r.id === sel} onClick={() => setSel(r.id)}>
              <span className="tid">{r.test_id}{r.rep > 0 ? ` ·r${r.rep}` : ''}</span>
              {r.passed !== null
                ? <PassPill passed={r.passed} />
                : <LabelPill label={r.label} />}
              {r.judge_label && r.judge_label !== r.label && (
                <span className="pill neutral" title={`judge disagrees: ${r.judge_label}`}>judge≠</span>
              )}
            </button>
          ))}
          {rows?.length === 0 && (
            <Empty title="Nothing matches">Clear a filter or search for something else.</Empty>
          )}
        </div>

        <div className="panel transcript">
          {selected ? <Transcript r={selected} /> : (
            <Empty title="No transcript selected">Pick a result on the left; arrow keys move the selection.</Empty>
          )}
        </div>
      </div>
    </section>
  )
}

function Transcript({ r }: { r: ResultRow }) {
  return (
    <div className="fade-in" key={r.id}>
      <div className="block">
        <div className="verdict-row">
          <span className="mono" style={{ fontWeight: 600 }}>{r.test_id}</span>
          <span className="pill neutral">{r.category}</span>
          {r.passed !== null && <PassPill passed={r.passed} />}
          {r.label && <LabelPill label={r.label} />}
          {r.flags && <span className="pill fail">{r.flags}</span>}
        </div>
        <div className="metric-row" style={{ marginTop: 8 }}>
          {r.latency_ms != null && <span className="num">{(r.latency_ms / 1000).toFixed(1)}s</span>}
          {r.tok_per_sec != null && <span className="num">{r.tok_per_sec.toFixed(0)} tok/s</span>}
          {r.prompt_tokens != null && <span className="num">{r.prompt_tokens} in / {r.completion_tokens} out</span>}
        </div>
      </div>

      {r.prompt_text && (
        <div className="block">
          <div className="block-label">PROMPT</div>
          <pre>{r.prompt_text}</pre>
        </div>
      )}

      <div className="block">
        <div className="block-label">RESPONSE</div>
        <pre>{r.response ?? '(empty response)'}</pre>
      </div>

      {r.detail && (
        <div className="block">
          <div className="block-label">GRADER DETAIL</div>
          <pre>{r.detail}</pre>
        </div>
      )}

      {(r.label || r.judge_label || r.human_label) && (
        <div className="block">
          <div className="block-label">VERDICTS</div>
          <div className="verdict-row">
            {r.label && <span>keyword: <LabelPill label={r.label} /></span>}
            {r.judge_label && (
              <span title={r.judge_reason ?? undefined}>
                judge ({r.judge_model}): <LabelPill label={r.judge_label} />
              </span>
            )}
            {r.human_label && <span>human: <LabelPill label={r.human_label} /></span>}
          </div>
          {r.judge_reason && (
            <p style={{ margin: '8px 0 0', color: 'var(--ink-mid)', fontSize: '0.82rem' }}>{r.judge_reason}</p>
          )}
        </div>
      )}
    </div>
  )
}
