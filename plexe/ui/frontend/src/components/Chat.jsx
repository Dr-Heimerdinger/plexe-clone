import React, { useEffect, useRef, useState } from 'react'

function Message({ msg }) {
    const isUser = msg.role === 'user'
    const isThinking = msg.role === 'thinking'

    if (isThinking) {
        return (
            <div className="message thinking">
                <div className="thinking-bubble">
                    <div className="thinking-header">
                        <span className="agent-name">{msg.agent_name}</span>
                        <span className="step-number">Step {msg.step_number}</span>
                    </div>
                    <div className="thinking-content">{msg.message}</div>
                </div>
            </div>
        )
    }

    return (
        <div className={`message ${isUser ? 'user' : 'assistant'}`}>
            <div className="bubble">{msg.content}</div>
        </div>
    )
}

function ThinkingIndicator() {
    return (
        <div className="message assistant">
            <div className="bubble thinking-indicator">
                <div className="dot-typing">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    )
}

export default function Chat({ messages, status, isProcessing, onSendMessage }) {
    const [input, setInput] = useState('')
    const messagesEndRef = useRef(null)

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isProcessing])

    const send = () => {
        if (onSendMessage(input)) {
            setInput('')
        }
    }

    const getStatusClass = () => {
        if (isProcessing) return 'processing'
        return status
    }

    const getStatusText = () => {
        if (isProcessing) return 'Processing...'
        if (status === 'connected') return 'Connected'
        if (status === 'disconnected') return 'Disconnected - Reconnecting...'
        if (status === 'error') return 'Connection Error'
        return status
    }

    return (
        <div className="chat-root">
            <div className={`status ${getStatusClass()}`}>{getStatusText()}</div>
            <div className="messages">
                {messages.map((m, i) => (
                    <Message key={i} msg={m} />
                ))}
                {isProcessing && <ThinkingIndicator />}
                <div ref={messagesEndRef} />
            </div>
            <div className="composer">
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                            send()
                        }
                    }}
                    placeholder="Type your message..."
                    disabled={isProcessing || status !== 'connected'}
                />
                <button onClick={send} disabled={isProcessing || status !== 'connected'}>
                    {isProcessing ? 'Processing...' : 'Send'}
                </button>
            </div>
        </div>
    )
}
