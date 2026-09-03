import { useState } from 'react'
import { uploadDocument } from '../services/api'

function UploadDocument({ onUploadSuccess }) {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState(null)
  const [isUploading, setIsUploading] = useState(false)

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0] || null
    setFile(selectedFile)
    setStatus(null)
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (!file) {
      setStatus({ type: 'error', message: 'Please select a PDF file.' })
      return
    }

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setStatus({ type: 'error', message: 'Please select a PDF file.' })
      return
    }

    setIsUploading(true)
    setStatus({ type: 'loading', message: 'Uploading and indexing...' })

    try {
      const result = await uploadDocument(file)
      setStatus({ type: 'success', result })
      onUploadSuccess()
    } catch (error) {
      setStatus({ type: 'error', message: error.message, requestId: error.requestId })
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <form className="upload-panel" onSubmit={handleSubmit}>
      <div className="upload-dropzone">
        <span className="upload-icon" aria-hidden="true">↑</span>
        <label htmlFor="document-upload">Upload PDF Document</label>
        <p>Drag and drop your PDF here<br />or click to browse files</p>
        <input
          id="document-upload"
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileChange}
          disabled={isUploading}
        />
        <label className="choose-file-button" htmlFor="document-upload">Choose PDF File</label>
        <small>PDF only</small>
        {file && <strong className="selected-file">{file.name}</strong>}
      </div>
      <button className="upload-button" type="submit" disabled={isUploading || !file}>{isUploading ? 'Indexing...' : 'Upload PDF'}</button>
      <div className="upload-status" aria-live="polite">
        {status?.type === 'loading' && <p role="status">{status.message} This may take a few seconds.</p>}
        {status?.type === 'success' && (
          <p role="status"><strong>PDF indexed successfully.</strong><br />{status.result.filename}<br />{status.result.pages} pages · {status.result.chunks} chunks · {status.result.status}</p>
        )}
        {status?.type === 'error' && (
          <p role="alert"><strong>Upload failed.</strong><br />{status.message}{status.requestId && <><br />Request ID: {status.requestId}</>}</p>
        )}
      </div>
    </form>
  )
}

export default UploadDocument
