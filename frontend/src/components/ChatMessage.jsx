import CitationList from './CitationList'

function ChatMessage({ message }) {
  return (
    <article className={`chat-message ${message.role}`}>
      <p className="message-role">{message.role === 'user' ? 'You' : 'Assistant'}</p>
      <p className="message-content">{message.content}</p>
      {message.role === 'assistant' && <CitationList citations={message.citations} />}
    </article>
  )
}

export default ChatMessage
