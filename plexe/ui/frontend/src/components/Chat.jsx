import React, { useEffect, useRef, useState } from 'react'

function Message({ msg }) {
    const isUser = msg.role === 'user'
    return (
        <div className={`message ${isUser ? 'user' : 'assistant'}`}>
            <div className="bubble">{msg.content}</div>
        </div>
    )
}

export default function Chat({ wsUrl }) {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [status, setStatus] = useState('disconnected')
    const wsRef = useRef(null)

    useEffect(() => {
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws

        ws.onopen = () => setStatus('connected')
        ws.onclose = () => setStatus('disconnected')
        ws.onmessage = (ev) => {
            try {
                const data = JSON.parse(ev.data)
                setMessages((m) => [...m, data])
            } catch (e) {
                console.error('invalid ws message', e)
            }
        }

        return () => ws.close()
    }, [wsUrl])

    const send = () => {
        if (!input.trim()) return
        const msg = { role: 'user', content: input }
        setMessages((m) => [...m, msg])
        wsRef.current?.send(JSON.stringify({ content: input }))
        setInput('')
    }

    return (
        <div className="chat-root">
            <div className="status">Status: {status}</div>
            <div className="messages">
                {messages.map((m, i) => (
                    <Message key={i} msg={m} />
                ))}
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
                />
                <button onClick={send}>Send</button>
            </div>
        </div>
    )
}
