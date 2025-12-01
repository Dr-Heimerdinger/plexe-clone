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

export default function Chat({ messages, status, onSendMessage, isProcessing }) {
    const [input, setInput] = useState('')
    const messagesEndRef = useRef(null)

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isProcessing])

    const send = () => {
        if (!input.trim()) return
        onSendMessage(input)
        setInput('')
    }

    return (
        <div className="chat-root">
            <div className="status">Status: {status}</div>
            <div className="messages">
                {messages.map((m, i) => (
                    <Message key={i} msg={m} />
                ))}
                {isProcessing && (
                    <div className="message thinking-indicator">
                        <div className="bubble processing">
                            <span className="dot"></span>
                            <span className="dot"></span>
                            <span className="dot"></span>
                        </div>
                    </div>
                )}
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
                    disabled={isProcessing}
                />
                <button onClick={send} disabled={isProcessing}>Send</button>
            </div>
        </div>
    )
}
