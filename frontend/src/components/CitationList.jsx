function CitationList({ citations = [] }) {
  if (!citations.length) return null

  return (
    <aside className="citations" aria-label="Sources">
      <h4>Sources</h4>
      <ul>
        {citations.map((citation) => (
          <li key={citation.citation_id}>
            <span>[{citation.citation_id}]</span> {citation.source} — Page {citation.page_label || citation.page}
          </li>
        ))}
      </ul>
    </aside>
  )
}

export default CitationList
