import { useState } from 'react'
import { deleteDocument } from '../services/api'

function DocumentCard({ document, onDeleteSuccess }) {
  const [isConfirming, setIsConfirming] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [status, setStatus] = useState(null)

  async function handleDelete() {
    setIsDeleting(true)
    setStatus(null)
    try {
      await deleteDocument(document.document_id)
      setStatus({ type: 'success', message: `${document.filename} deleted successfully.` })
      setIsConfirming(false)
      window.setTimeout(onDeleteSuccess, 500)
    } catch (error) {
      setStatus({ type: 'error', message: error.message, requestId: error.requestId })
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <article className="document-card" aria-label={`Document: ${document.filename}`}>
      <h3 title={document.filename}>{document.filename}</h3>
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
      {!isConfirming && !isDeleting && status?.type !== 'success' && (
        <button className="delete-button" type="button" onClick={() => setIsConfirming(true)} aria-label={`Delete ${document.filename}`}>
          Delete
        </button>
      )}
      {isConfirming && (
        <div className="delete-confirmation" role="group" aria-label={`Confirm deletion of ${document.filename}`}>
          <p>Delete {document.filename}?</p>
          <button type="button" onClick={() => setIsConfirming(false)} disabled={isDeleting}>Cancel</button>
          <button className="delete-button" type="button" onClick={handleDelete} disabled={isDeleting}>{isDeleting ? 'Deleting...' : 'Delete'}</button>
        </div>
      )}
      {status?.type === 'success' && <p className="delete-success" role="status">{status.message}</p>}
      {status?.type === 'error' && <p className="delete-error" role="alert"><strong>Unable to delete document.</strong><br />{status.message}{status.requestId && <><br />Request ID: {status.requestId}</>}</p>}
    </article>
  )
}

export default DocumentCard
