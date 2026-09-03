import { useEffect, useState } from 'react'
import { getHealth, getReadiness } from './services/api'
import DocumentList from './components/DocumentList'
import UploadDocument from './components/UploadDocument'
import ChatPanel from './components/ChatPanel'
import Sidebar from './components/Sidebar'
import Header from './components/Header'

function App() {
  const [documentRefreshKey, setDocumentRefreshKey] = useState(0)
  const [theme, setTheme] = useState(() => localStorage.getItem('enterprise-rag-theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'))
  const [systemStatus, setSystemStatus] = useState('Checking status...')
  const [documentCount, setDocumentCount] = useState(null)
  const refreshDocuments = () => setDocumentRefreshKey((key) => key + 1)

  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('enterprise-rag-theme', theme) }, [theme])
  useEffect(() => { Promise.all([getHealth(), getReadiness()]).then(() => setSystemStatus('Ready')).catch(() => setSystemStatus('Backend unavailable')) }, [])

  return <div className="app-layout">
    <Sidebar systemStatus={systemStatus} />
    <main className="app-shell">
      <Header theme={theme} systemStatus={systemStatus} onThemeToggle={() => setTheme((current) => current === 'light' ? 'dark' : 'light')} />
      <section className="workspace" aria-label="Assistant workspace">
        <div className="documents-area" id="documents"><UploadDocument onUploadSuccess={refreshDocuments} /><DocumentList refreshKey={documentRefreshKey} onDeleteSuccess={refreshDocuments} onDocumentsLoaded={setDocumentCount} /></div>
        <div id="chat"><ChatPanel documentCount={documentCount ?? 0} /></div>
      </section>
    </main>
  </div>
}
export default App
