import { useEffect, useState } from 'react'
import { getDocuments } from '../services/api'
import DocumentCard from './DocumentCard'

function DocumentList() {
  const [data, setData] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let isMounted = true

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
  }, [])

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
    <section className="content-card documents-section" aria-labelledby="documents-heading">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Indexed knowledge</p>
          <h2 id="documents-heading">Documents</h2>
        </div>
        <div className="summary" aria-label="Document totals">
          <span><strong>{data?.total_documents ?? documents.length}</strong> indexed documents</span>
          <span><strong>{data?.total_chunks ?? 0}</strong> total chunks</span>
        </div>
      </div>

      {documents.length === 0 ? (
        <p className="empty-state">No documents have been indexed yet.</p>
      ) : (
        <div className="document-grid">
          {documents.map((document) => <DocumentCard key={document.document_id} document={document} />)}
        </div>
      )}
    </section>
  )
}

export default DocumentList
