import { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, Loader2, Sparkles, HelpCircle, ChevronDown, ChevronUp } from 'lucide-react'
import { apiFetch } from '../utils/api'


export default function Chat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "👋 Hi! I'm the Energy AI Assistant. Ask me anything about this platform — its data, ML models, analysis, or recommendations!",
      sources: []
    }
  ])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [expandedSources, setExpandedSources] = useState({})
  
  const chatEndRef = useRef(null)

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, sending])

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || sending) return

    const userMsg = input.trim()
    setInput('')
    setSending(true)

    // Add user message to state
    setMessages((prev) => [...prev, { role: 'user', content: userMsg, sources: [] }])

    try {
      // Map message history into the API's simple format
      const history = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }))

      const res = await apiFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg, chat_history: history })
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to fetch chatbot answer')
      }

      const data = await res.json()

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          sources: data.sources || []
        }
      ])
    } catch (e) {
      console.error(e)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ Sorry, I ran into an error: ${e.message}`,
          sources: []
        }
      ])
    } finally {
      setSending(false)
    }
  }

  const toggleSources = (index) => {
    setExpandedSources(prev => ({
      ...prev,
      [index]: !prev[index]
    }))
  }

  const samplePrompts = [
    "What models are used for forecasting?",
    "How does the anomaly detection work?",
    "What is the R² score of Random Forest?",
    "What are the demographics in London?",
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 5rem)', gap: '1.5rem' }}>
      <div>
        <h2 style={{ fontSize: '2rem', color: '#fff', marginBottom: '0.25rem' }}>🤖 AI Energy Assistant</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Interact with the RAG chatbot to resolve doubts about datasets, models, algorithms, or insights.</p>
      </div>

      <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)' }}></div>

      {/* Chat Messages Frame */}
      <div className="glass-panel" style={{ 
        flex: 1, 
        display: 'flex', 
        flexDirection: 'column', 
        overflow: 'hidden', 
        borderRadius: '16px',
        background: 'rgba(15, 23, 42, 0.2)' 
      }}>
        {/* Messages Body */}
        <div style={{ 
          flex: 1, 
          overflowY: 'auto', 
          padding: '1.5rem', 
          display: 'flex', 
          flexDirection: 'column', 
          gap: '1.25rem' 
        }}>
          {messages.map((msg, idx) => {
            const isUser = msg.role === 'user'
            return (
              <div key={idx} style={{ 
                display: 'flex', 
                justifyContent: isUser ? 'flex-end' : 'flex-start',
                width: '100%',
                animation: 'fadeIn 0.25s ease-out forwards'
              }}>
                <div style={{ display: 'flex', gap: '0.75rem', maxWidth: '80%', flexDirection: isUser ? 'row-reverse' : 'row' }}>
                  {/* Icon Avatar */}
                  <div style={{
                    background: isUser ? 'var(--primary-glow)' : 'var(--secondary-glow)',
                    color: isUser ? 'var(--primary)' : 'var(--secondary)',
                    border: isUser ? '1px solid rgba(0, 212, 170, 0.2)' : '1px solid rgba(124, 58, 237, 0.2)',
                    width: '36px',
                    height: '36px',
                    borderRadius: '10px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0
                  }}>
                    {isUser ? <User size={18} /> : <Bot size={18} />}
                  </div>

                  {/* Message Bubble */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <div style={{ 
                      background: isUser ? 'linear-gradient(135deg, rgba(0, 212, 170, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%)' : 'rgba(30, 41, 59, 0.4)',
                      border: isUser ? '1px solid rgba(0, 212, 170, 0.25)' : '1px solid var(--card-border)',
                      borderRadius: isUser ? '16px 4px 16px 16px' : '4px 16px 16px 16px',
                      padding: '0.9rem 1.25rem',
                      color: '#fff',
                      fontSize: '0.95rem',
                      lineHeight: 1.6,
                      whiteSpace: 'pre-wrap',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                    }}>
                      {msg.content}
                    </div>

                    {/* Sources Expander */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div style={{ marginTop: '0.25rem' }}>
                        <button 
                          onClick={() => toggleSources(idx)}
                          style={{ 
                            background: 'none', 
                            border: 'none', 
                            cursor: 'pointer', 
                            color: 'var(--text-muted)', 
                            fontSize: '0.75rem', 
                            fontWeight: 600,
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '0.25rem',
                            padding: '0.25rem'
                          }}
                        >
                          <span>📚 Sources ({msg.sources.length})</span>
                          {expandedSources[idx] ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </button>

                        {expandedSources[idx] && (
                          <div style={{ 
                            display: 'flex', 
                            flexDirection: 'column', 
                            gap: '0.5rem', 
                            marginTop: '0.5rem', 
                            padding: '0.75rem', 
                            background: 'rgba(255,255,255,0.01)', 
                            border: '1px solid var(--card-border)', 
                            borderRadius: '8px' 
                          }}>
                            {msg.sources.map((src, sIdx) => (
                              <div key={sIdx} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                <span style={{ color: 'var(--primary)', fontWeight: 700 }}>[{src.source}]</span>
                                <pre style={{ 
                                  whiteSpace: 'pre-wrap', 
                                  fontFamily: 'monospace', 
                                  fontSize: '0.75rem', 
                                  color: 'var(--text-muted)', 
                                  background: 'rgba(0,0,0,0.1)', 
                                  padding: '0.5rem', 
                                  borderRadius: '4px',
                                  marginTop: '0.25rem',
                                  border: '1px solid rgba(255,255,255,0.02)'
                                }}>
                                  {src.content.substring(0, 200)}...
                                </pre>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
          {sending && (
            <div style={{ display: 'flex', justifyContent: 'flex-start', width: '100%' }}>
              <div style={{ display: 'flex', gap: '0.75rem', maxWidth: '80%' }}>
                <div style={{
                  background: 'var(--secondary-glow)',
                  color: 'var(--secondary)',
                  border: '1px solid rgba(124, 58, 237, 0.2)',
                  width: '36px',
                  height: '36px',
                  borderRadius: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Loader2 className="animate-spin" size={18} />
                </div>
                <div style={{
                  background: 'rgba(30, 41, 59, 0.4)',
                  border: '1px solid var(--card-border)',
                  borderRadius: '4px 16px 16px 16px',
                  padding: '0.9rem 1.25rem',
                  color: 'var(--text-secondary)',
                  fontSize: '0.95rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  <span>Querying knowledge base...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Suggested Prompts (only shown initially) */}
        {messages.length === 1 && (
          <div style={{ padding: '0 1.5rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', trackingLetter: '0.05em', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <Sparkles size={12} color="var(--primary)" /> Suggested Queries
            </span>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
              {samplePrompts.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => setInput(p)}
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid var(--card-border)',
                    borderRadius: '99px',
                    padding: '0.4rem 1rem',
                    color: 'var(--text-secondary)',
                    fontSize: '0.8rem',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                  className="prompt-chip"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Bar */}
        <form onSubmit={handleSend} style={{ 
          padding: '1.25rem', 
          borderTop: '1px solid var(--card-border)', 
          background: 'rgba(9, 11, 17, 0.6)',
          display: 'flex', 
          gap: '0.75rem' 
        }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={sending}
            placeholder="Ask about model comparisons, PCA anomalies, weather correlations..."
            style={{
              flex: 1,
              background: 'rgba(15, 23, 42, 0.4)',
              border: '1px solid var(--card-border)',
              borderRadius: '12px',
              padding: '0.75rem 1.25rem',
              color: '#fff',
              fontSize: '0.95rem',
              outline: 'none'
            }}
          />
          <button 
            type="submit" 
            disabled={sending || !input.trim()}
            style={{
              background: 'var(--primary)',
              color: '#042f2e',
              border: 'none',
              borderRadius: '12px',
              width: '45px',
              height: '45px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              opacity: (sending || !input.trim()) ? 0.5 : 1
            }}
          >
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  )
}
