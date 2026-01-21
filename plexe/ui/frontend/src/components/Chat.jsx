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

function ConfirmationDialog({ request, onConfirm, onReject }) {
    const [isExpanded, setIsExpanded] = useState(false)

    const renderContent = () => {
        const { content, content_type } = request

        // Truncate content if too long and not expanded
        const maxPreviewLength = 500
        const shouldTruncate = content.length > maxPreviewLength && !isExpanded
        const displayContent = shouldTruncate
            ? content.substring(0, maxPreviewLength) + '...'
            : content

        if (content_type === 'code') {
            return (
                <pre className="confirmation-code">
                    <code>{displayContent}</code>
                </pre>
            )
        } else if (content_type === 'json') {
            try {
                const parsed = JSON.parse(content)
                return (
                    <pre className="confirmation-json">
                        {isExpanded ? JSON.stringify(parsed, null, 2) : JSON.stringify(parsed, null, 2).substring(0, maxPreviewLength) + (JSON.stringify(parsed, null, 2).length > maxPreviewLength ? '...' : '')}
                    </pre>
                )
            } catch {
                return <pre className="confirmation-text">{displayContent}</pre>
            }
        } else if (content_type === 'markdown') {
            // Simple markdown rendering (just preserve formatting)
            return <div className="confirmation-markdown"><pre>{displayContent}</pre></div>
        }
        return <div className="confirmation-text"><pre>{displayContent}</pre></div>
    }

    return (
        <div className="confirmation-dialog-overlay">
            <div className="confirmation-dialog">
                <div className="confirmation-header">
                    <h3>{request.title}</h3>
                </div>
                <div className="confirmation-body">
                    {renderContent()}
                    {request.content.length > 500 && (
                        <button
                            className="expand-toggle"
                            onClick={() => setIsExpanded(!isExpanded)}
                        >
                            {isExpanded ? '▲ Thu gọn' : '▼ Xem thêm'}
                        </button>
                    )}
                </div>
                <div className="confirmation-footer">
                    <button className="btn-reject" onClick={onReject}>
                        ✗ Reject
                    </button>
                    <button className="btn-confirm" onClick={onConfirm}>
                        ✓ Confirm
                    </button>
                </div>
            </div>
        </div>
    )
}

export default function Chat({ messages, status, isProcessing, onSendMessage, onStopProcessing, confirmationRequest, onConfirmationResponse }) {
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
            {confirmationRequest && (
                <ConfirmationDialog
                    request={confirmationRequest}
                    onConfirm={() => onConfirmationResponse(confirmationRequest.id, true)}
                    onReject={() => onConfirmationResponse(confirmationRequest.id, false)}
                />
            )}
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
                {isProcessing ? (
                    <button onClick={onStopProcessing} className="stop-btn">
                        ⬛ Stop
                    </button>
                ) : (
                    <button onClick={send} disabled={status !== 'connected'}>
                        Send
                    </button>
                )}
            </div>
        </div>
    )
}
