import { useState } from 'react'
import { sendChat } from '../services/api'
import ChatMessage from './ChatMessage'

function ChatPanel() {
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
        {!messages.length && !isThinking && <p className="empty-state">Ask questions about your indexed documents.</p>}
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
