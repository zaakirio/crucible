import { useEffect, useRef, useState } from 'react'

interface Msg { role: 'user' | 'assistant'; content: string }

const LS_KEY = 'crucible-lab-playground'

export default function Playground() {
  const [server, setServer] = useState('http://127.0.0.1:8080/v1')
  const [model, setModel] = useState('default')
  const [temp, setTemp] = useState(0.7)
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const saved = localStorage.getItem(LS_KEY)
    if (saved) {
      try {
        const s = JSON.parse(saved)
        if (s.server) setServer(s.server)
        if (s.model) setModel(s.model)
      } catch { /* stale localStorage is fine to ignore */ }
    }
  }, [])

  useEffect(() => {
    localStorage.setItem(LS_KEY, JSON.stringify({ server, model }))
  }, [server, model])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [msgs])

  // Abort an in-flight stream when the view unmounts.
  useEffect(() => () => abortRef.current?.abort(), [])

  async function send() {
    const text = input.trim()
    if (!text || busy) return
    setError('')
    setInput('')
    const history: Msg[] = [...msgs, { role: 'user', content: text }]
    setMsgs([...history, { role: 'assistant', content: '' }])
    setBusy(true)

    const ac = new AbortController()
    abortRef.current = ac
    try {
      const res = await fetch('/api/playground/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ server, model, temperature: temp, messages: history }),
        signal: ac.signal,
      })
      if (!res.ok || !res.body) throw new Error(`${res.status} ${await res.text()}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') continue
          let parsed: { error?: string; choices?: { delta?: { content?: string } }[] }
          try { parsed = JSON.parse(data) } catch { continue }
          if (parsed.error) throw new Error(parsed.error)
          const delta = parsed.choices?.[0]?.delta?.content
          if (delta) {
            setMsgs(m => {
              const copy = m.slice()
              copy[copy.length - 1] = {
                role: 'assistant',
                content: copy[copy.length - 1].content + delta,
              }
              return copy
            })
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError(String((e as Error).message ?? e))
      setMsgs(m => (m.length && m[m.length - 1].content === '' ? m.slice(0, -1) : m))
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  return (
    <div className="fade-in">
      <div className="page-head">
        <h1>Playground</h1>
        <div className="spacer" />
        <div className="controls">
          <input className="input mono" value={server} onChange={e => setServer(e.target.value)}
            aria-label="OpenAI-compatible server URL" style={{ width: 250 }} />
          <input className="input mono" value={model} onChange={e => setModel(e.target.value)}
            aria-label="model name" style={{ width: 150 }} />
          <label className="chip" style={{ gap: 8 }}>
            temp
            <input type="range" min={0} max={2} step={0.1} value={temp}
              onChange={e => setTemp(Number(e.target.value))} aria-label="temperature" />
            <b className="num">{temp.toFixed(1)}</b>
          </label>
          <button className="btn" onClick={() => { setMsgs([]); setError('') }} disabled={msgs.length === 0}>
            Clear
          </button>
        </div>
      </div>

      <p style={{ color: 'var(--ink-muted)', marginTop: 0, maxWidth: '75ch' }}>
        Talks to any OpenAI-compatible server through the lab process (llama-server, Ollama, LM Studio, vLLM).
        Point it at the same server crucible evaluates so what you probe is what got scored.
      </p>

      {error && <div className="error-note" role="alert" style={{ marginBottom: 12 }}>{error}</div>}

      <div className="chat" aria-live="polite">
        {msgs.length === 0 && (
          <div className="empty" style={{ textAlign: 'left', padding: '24px 0' }}>
            <h3>No conversation yet</h3>
            <p style={{ margin: 0 }}>
              Start llama-server (<code>./scripts/run-model.sh</code> in inf-eng, or any OpenAI-compatible server),
              set the URL above, and ask something.
            </p>
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <span className="who">{m.role === 'user' ? 'you' : model}</span>
            <div className={`bubble${busy && i === msgs.length - 1 && m.role === 'assistant' ? ' streaming' : ''}`}>
              {m.content || (busy && i === msgs.length - 1 ? '' : '(empty)')}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form className="chat-form" onSubmit={e => { e.preventDefault(); send() }}>
        <textarea className="input" value={input} placeholder="Message the model… (Enter to send, Shift+Enter for newline)"
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
          }}
          aria-label="message" />
        {busy ? (
          <button type="button" className="btn" onClick={() => abortRef.current?.abort()}>Stop</button>
        ) : (
          <button type="submit" className="btn primary" disabled={!input.trim()}>Send</button>
        )}
      </form>
    </div>
  )
}
