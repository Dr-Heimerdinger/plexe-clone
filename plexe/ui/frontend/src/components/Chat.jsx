import React, { useEffect, useRef, useState } from 'react'

// Icons for different event types
const EventIcon = ({ eventType }) => {
    const icons = {
        agent_start: '🚀',
        thinking: '💭',
        tool_call: '🔧',
        tool_result: '✅',
        agent_end: '🎯'
    }
    return <span className="event-icon">{icons[eventType] || '•'}</span>
}

// Specific color mapping per agent for consistent identity visuals
const getAgentTheme = (name = '') => {
    const lowerName = name.toLowerCase()

    // Specific colors for known agents
    if (lowerName.includes('orchestrator')) {
        return {
            accent: '#2563eb',
            tint: 'rgba(37, 99, 235, 0.08)',
            wash: 'rgba(37, 99, 235, 0.04)',
            border: 'rgba(37, 99, 235, 0.3)'
        }
    }

    if (lowerName.includes('conversational')) {
        return {
            accent: '#10b981',
            tint: 'rgba(16, 185, 129, 0.08)',
            wash: 'rgba(16, 185, 129, 0.04)',
            border: 'rgba(16, 185, 129, 0.3)'
        }
    }

    // Fallback to deterministic color for other agents
    let hash = 0
    for (let i = 0; i < name.length; i += 1) {
        hash = name.charCodeAt(i) + ((hash << 5) - hash)
    }
    const hue = Math.abs(hash) % 360
    return {
        accent: `hsl(${hue}, 70%, 50%)`,
        tint: `hsla(${hue}, 70%, 50%, 0.08)`,
        wash: `hsla(${hue}, 70%, 50%, 0.04)`,
        border: `hsla(${hue}, 70%, 50%, 0.3)`
    }
}

// Group consecutive messages from the same agent
const groupMessagesByAgent = (messages) => {
    const groups = []
    let currentGroup = null

    messages.forEach((msg) => {
        if (msg.role === 'thinking') {
            const agentName = msg.agent_name || 'Agent'

            // Start a new group if agent changes or no current group
            if (!currentGroup || currentGroup.agent !== agentName) {
                if (currentGroup) {
                    groups.push(currentGroup)
                }
                currentGroup = {
                    agent: agentName,
                    steps: [],
                    startStep: msg.step_number,
                    endStep: msg.step_number
                }
            }

            // Add step to current group
            currentGroup.steps.push(msg)
            currentGroup.endStep = msg.step_number
        } else {
            // Non-thinking messages close the current group
            if (currentGroup) {
                groups.push(currentGroup)
                currentGroup = null
            }
            groups.push({ type: 'message', message: msg })
        }
    })

    // Add the last group if exists
    if (currentGroup) {
        groups.push(currentGroup)
    }

    return groups
}

// Render a single event within an agent group
const EventItem = ({ step }) => {
    const eventType = step.event_type || 'thinking'
    const hasError = step.message && step.message.toLowerCase().includes('error')

    const renderMessage = () => {
        if (!step.message) return null

        if (hasError) {
            const parts = step.message.split(/(Error:?[^\n]*|error:?[^\n]*)/gi)
            return (
                <>
                    {parts.map((part, i) => {
                        if (part.toLowerCase().includes('error')) {
                            return <span key={i} className="error-text">{part}</span>
                        }
                        return <span key={i}>{part}</span>
                    })}
                </>
            )
        }

        return step.message
    }

    return (
        <div className={`event-item ${eventType} ${hasError ? 'has-error' : ''}`}>
            <div className="event-header">
                <EventIcon eventType={eventType} />
                <span className="event-label">
                    {eventType === 'agent_start' && 'Starting'}
                    {eventType === 'thinking' && 'Reasoning'}
                    {eventType === 'tool_call' && `Tool: ${step.tool_name || 'Unknown'}`}
                    {eventType === 'tool_result' && 'Result'}
                    {eventType === 'agent_end' && 'Completed'}
                </span>
                {step.timestamp && (
                    <span className="event-time">{step.timestamp}</span>
                )}
            </div>
            <div className="event-content">
                {renderMessage()}
            </div>
        </div>
    )
}

// Render an agent group (multiple steps from same agent)
const AgentGroup = ({ group }) => {
    const theme = getAgentTheme(group.agent)
    const hasError = group.steps.some(s => s.message && s.message.toLowerCase().includes('error'))

    const stepRange = group.startStep === group.endStep
        ? `Step ${group.startStep}`
        : `Steps ${group.startStep}-${group.endStep}`

    return (
        <div className="agent-group">
            <div
                className={`agent-group-bubble ${hasError ? 'has-error' : ''}`}
                style={{
                    borderLeftColor: hasError ? '#ef4444' : theme.accent,
                    borderColor: hasError ? 'rgba(239, 68, 68, 0.3)' : theme.border
                }}
            >
                <div className="agent-group-header" style={{
                    background: hasError
                        ? 'linear-gradient(135deg, rgba(254, 242, 242, 0.3) 0%, rgba(254, 226, 226, 0.2) 100%)'
                        : `linear-gradient(135deg, ${theme.wash} 0%, ${theme.tint} 100%)`
                }}>
                    <span className="agent-name" style={{ color: theme.accent }}>
                        {group.agent}
                    </span>
                    <span className="step-range" style={{
                        color: theme.accent,
                        opacity: 0.7
                    }}>
                        {stepRange}
                    </span>
                </div>
                <div className="agent-group-content">
                    {group.steps.map((step, idx) => (
                        <EventItem key={idx} step={step} />
                    ))}
                </div>
            </div>
        </div>
    )
}

function Message({ msg }) {
    const isUser = msg.role === 'user'
    const isError = msg.role === 'error'

    return (
        <div className={`message ${isUser ? 'user' : isError ? 'error' : 'assistant'}`}>
            <div className={`bubble ${isError ? 'error' : ''}`}>{msg.content}</div>
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
                            {isExpanded ? ' Compact' : ' See more'}
                        </button>
                    )}
                </div>
                <div className="confirmation-footer">
                    <button className="btn-reject" onClick={onReject}>
                        Reject
                    </button>
                    <button className="btn-confirm" onClick={onConfirm}>
                        Confirm
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

    // Group messages by agent
    const messageGroups = groupMessagesByAgent(messages)

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
                {messageGroups.map((group, i) => {
                    if (group.type === 'message') {
                        return <Message key={i} msg={group.message} />
                    } else {
                        return <AgentGroup key={i} group={group} />
                    }
                })}
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
                        Stop
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
