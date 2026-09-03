import { useEffect, useState } from 'react'
import { getDocuments } from '../services/api'
import DocumentCard from './DocumentCard'

function DocumentList({ refreshKey = 0, onDeleteSuccess }) {
  const [data, setData] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let isMounted = true

    setIsLoading(true)
    setError(null)

    getDocuments()
      .then((documents) => {
        if (isMounted) setData(documents)
      })
      .catch((requestError) => {
        if (isMounted) setError(requestError)
      })
      .finally(() => {
        if (isMounted) setIsLoading(false)
      })

    return () => { isMounted = false }
  }, [refreshKey])

  if (isLoading) return <section className="content-card"><h2>Documents</h2><p role="status">Loading indexed documents...</p></section>

  if (error) {
    return (
      <section className="content-card" role="alert">
        <h2>Documents</h2>
        <p>Unable to load documents.</p>
        <p className="error-detail">{error.message}</p>
        {error.requestId && <p className="request-id">Request ID: {error.requestId}</p>}
      </section>
    )
  }

  const documents = data?.documents || []

  return (
    <>
    <section className="content-card knowledge-stats" aria-labelledby="knowledge-heading">
      <div className="section-heading"><div><p className="section-kicker">Knowledge base</p><h2 id="knowledge-heading">Indexed Knowledge</h2></div></div>
      <div className="summary" aria-label="Document totals"><span><strong>{data?.total_documents ?? documents.length}</strong>Indexed Documents</span><span><strong>{data?.total_chunks ?? 0}</strong>Total Chunks</span></div>
    </section>
    <section className="content-card documents-section" aria-labelledby="documents-heading">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Indexed knowledge</p>
          <h2 id="documents-heading">Documents</h2>
        </div>
      </div>

      {documents.length === 0 ? (
        <p className="empty-state">No documents have been indexed yet.</p>
      ) : (
        <div className="document-grid">
          {documents.map((document) => <DocumentCard key={document.document_id} document={document} onDeleteSuccess={onDeleteSuccess} />)}
        </div>
      )}
    </section>
    </>
  )
}

export default DocumentList
