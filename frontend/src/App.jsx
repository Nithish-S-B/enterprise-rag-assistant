import { API_BASE_URL } from './services/api'
import DocumentList from './components/DocumentList'

const apiBaseUrl = API_BASE_URL

function App() {
  return (
    <main className="app-shell">
      <header className="app-header">
        <p className="eyebrow">Enterprise knowledge workspace</p>
        <h1>Enterprise RAG Assistant</h1>
        <p className="backend-status">Backend: {apiBaseUrl}</p>
      </header>

      <section className="workspace" aria-label="Assistant workspace">
        <DocumentList />
        <article className="placeholder-card">
          <h2>Chat</h2>
          <p>Ask questions about your documents here.</p>
        </article>
      </section>
    </main>
  )
}

export default App
