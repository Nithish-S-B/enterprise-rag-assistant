const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

class ApiError extends Error {
  constructor({ message, errorType, requestId, status }) {
    super(message)
    this.name = 'ApiError'
    this.errorType = errorType
    this.requestId = requestId
    this.status = status
  }
}

async function request(path, options = {}) {
  let response

  try {
    response = await fetch(`${API_BASE_URL}${path}`, options)
  } catch {
    throw new ApiError({
      message: 'Unable to connect to the backend.',
      errorType: 'network_error',
      requestId: undefined,
      status: undefined,
    })
  }

  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json')
    ? await response.json()
    : undefined

  if (!response.ok) {
    throw new ApiError({
      message: payload?.message || 'The backend request failed.',
      errorType: payload?.error_type || 'api_error',
      requestId: response.headers.get('X-Request-ID') || payload?.request_id,
      status: response.status,
    })
  }

  return payload
}

function getHealth() {
  return request('/api/health')
}

function getReadiness() {
  return request('/api/ready')
}

function getDocuments() {
  return request('/api/documents')
}

function uploadDocument(file) {
  if (!(file instanceof File)) {
    throw new TypeError('uploadDocument requires a File object.')
  }

  const formData = new FormData()
  formData.append('file', file)
  return request('/api/documents/upload', { method: 'POST', body: formData })
}

function deleteDocument(documentId) {
  return request(`/api/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' })
}

function sendChat(question, finalK = 4) {
  return request('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, final_k: finalK }),
  })
}

export { API_BASE_URL, ApiError, getHealth, getReadiness, getDocuments, uploadDocument, deleteDocument, sendChat }
