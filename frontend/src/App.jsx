const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function App() {
  return (
    <main className="app-shell">
      <header className="app-header">
        <p className="eyebrow">Enterprise knowledge workspace</p>
        <h1>Enterprise RAG Assistant</h1>
        <p className="backend-status">Backend: {apiBaseUrl}</p>
      </header>

      <section className="workspace" aria-label="Assistant workspace">
        <article className="placeholder-card">
          <h2>Documents</h2>
          <p>Document management will be available here.</p>
        </article>
        <article className="placeholder-card">
          <h2>Chat</h2>
          <p>Ask questions about your documents here.</p>
        </article>
      </section>
    </main>
  )
}

export default App
