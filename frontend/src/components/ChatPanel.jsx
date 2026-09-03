import { useState } from 'react'
import { sendChat } from '../services/api'
import ChatMessage from './ChatMessage'

function ChatPanel({ documentCount = 0 }) {
  const suggestions = ['Summarize the key points from my documents.', 'What are the most important policies or requirements?', 'What actions or responsibilities are mentioned?', 'What should I know from these documents?']
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [isThinking, setIsThinking] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(event) {
    event.preventDefault()
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion) {
      setError({ type: 'validation', message: 'Enter a question before sending.' })
      return
    }

    setError(null)
    setMessages((current) => [...current, { id: `${Date.now()}-user`, role: 'user', content: trimmedQuestion }])
    setIsThinking(true)

    try {
      const result = await sendChat(trimmedQuestion)
      setMessages((current) => [...current, {
        id: `${Date.now()}-assistant`,
        role: 'assistant',
        content: result.answer,
        citations: result.citations,
      }])
      setQuestion('')
    } catch (requestError) {
      setError({ ...requestError, type: 'api' })
    } finally {
      setIsThinking(false)
    }
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSubmit(event)
    }
  }

  return (
    <section className="chat-panel content-card" aria-labelledby="chat-heading">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Grounded answers</p>
          <h2 id="chat-heading">Chat</h2>
        </div>
      </div>
      <div className="conversation" aria-live="polite">
        {!messages.length && !isThinking && <div className="chat-empty"><img src="/logo.png" alt="" /><h3>Ask anything about your documents</h3><p>Get accurate, grounded answers with citations from your enterprise knowledge base.</p>{documentCount > 0 ? <div className="suggestions">{suggestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => setQuestion(suggestion)}>{suggestion}<span>→</span></button>)}</div> : <div className="no-documents"><strong>No documents are indexed yet.</strong><span>Upload a PDF document to start asking grounded questions.</span></div>}</div>}
        {messages.map((message) => <ChatMessage key={message.id} message={message} />)}
        {isThinking && <p className="thinking" role="status">Thinking...</p>}
      </div>
      {error && <div className="chat-error" role="alert">
        {error.type === 'validation' ? error.message : <><strong>Unable to generate an answer.</strong><br />{error.message}{error.requestId && <><br />Request ID: {error.requestId}</>}</>}
      </div>}
      <form className="chat-form" onSubmit={handleSubmit}>
        <label htmlFor="chat-question">Ask a question</label>
        <textarea id="chat-question" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={handleKeyDown} placeholder="Ask a question about your documents..." rows="3" disabled={isThinking} />
        <button type="submit" disabled={isThinking}>{isThinking ? 'Thinking...' : 'Send'}</button>
      </form>
    </section>
  )
}

export default ChatPanel
