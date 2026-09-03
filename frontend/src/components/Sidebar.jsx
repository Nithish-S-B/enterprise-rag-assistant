function Sidebar({ systemStatus }) {
  return <aside className="sidebar" aria-label="Primary navigation">
    <div className="brand-mark"><span aria-hidden="true" /><div><strong>Enterprise RAG</strong><small>Assistant</small></div></div>
    <nav><p>Workspace</p><a className="active" href="#documents">Workspace</a><a href="#documents">Documents</a><a href="#chat">Chat</a><p>Manage</p><a href="#settings">Settings</a><a href="#activity">Activity</a><a href="#api-keys">API Keys</a></nav>
    <div className="system-card"><small>System Status</small><strong><span className={`status-dot ${systemStatus === 'Ready' ? 'ready' : ''}`}>●</span> {systemStatus === 'Ready' ? 'All systems operational' : systemStatus}</strong></div>
  </aside>
}
export default Sidebar
