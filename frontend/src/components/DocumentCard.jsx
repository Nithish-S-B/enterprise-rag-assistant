function DocumentCard({ document }) {
  return (
    <article className="document-card" aria-label={`Document: ${document.filename}`}>
      <h3 title={document.filename}>{document.filename}</h3>
      <p className="document-id">Document ID: {document.document_id}</p>
      <dl className="document-stats">
        <div>
          <dt>Pages</dt>
          <dd>{document.pages}</dd>
        </div>
        <div>
          <dt>Chunks</dt>
          <dd>{document.chunks}</dd>
        </div>
      </dl>
    </article>
  )
}

export default DocumentCard
