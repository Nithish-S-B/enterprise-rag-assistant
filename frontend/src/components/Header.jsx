function Header({ theme, systemStatus, onThemeToggle }) {
  return <header className="app-header"><div><p className="eyebrow">Enterprise knowledge workspace</p><h1>Enterprise Knowledge Workspace</h1><p className="header-subtitle">Ask questions, get grounded answers from your enterprise documents.</p></div><div className="header-actions"><span className="readiness-pill"><span className={`status-dot ${systemStatus === 'Ready' ? 'ready' : ''}`}>●</span> {systemStatus === 'Ready' ? 'Ready' : systemStatus}</span><button className="theme-toggle" type="button" onClick={onThemeToggle} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}>{theme === 'light' ? 'Dark mode' : 'Light mode'}</button></div></header>
}
export default Header
