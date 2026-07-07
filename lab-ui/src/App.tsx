import { useEffect, useState } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import { api, type Meta } from './api'
import Compare from './views/Compare'
import Playground from './views/Playground'
import RunDetail from './views/RunDetail'
import Runs from './views/Runs'

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)

  useEffect(() => {
    api.meta().then(setMeta).catch(() => setMeta(null))
  }, [])

  return (
    <div className="shell">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <svg width="18" height="18" viewBox="0 0 32 32" aria-hidden="true">
            <path d="M6 4h20v6l-6 8v8a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-8L6 10z" fill="var(--accent)" />
          </svg>
          Crucible <span className="sub">Lab</span>
        </NavLink>
        <nav className="topnav">
          <NavLink to="/" end className={({ isActive }) => (isActive ? 'active' : '')}>Runs</NavLink>
          <NavLink to="/compare" className={({ isActive }) => (isActive ? 'active' : '')}>Diff</NavLink>
          <NavLink to="/playground" className={({ isActive }) => (isActive ? 'active' : '')}>Playground</NavLink>
        </nav>
        {meta && (
          <div className="meta num">
            <span>{meta.n_runs} runs</span>
            <span>{meta.n_results.toLocaleString()} results</span>
            <span>{meta.n_judged.toLocaleString()} judged</span>
            <span className="mono">{meta.db_path}</span>
          </div>
        )}
      </header>
      <main className="main">
        <Routes>
          <Route path="/" element={<Runs />} />
          <Route path="/runs/:id" element={<RunDetail />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/playground" element={<Playground />} />
        </Routes>
      </main>
    </div>
  )
}
