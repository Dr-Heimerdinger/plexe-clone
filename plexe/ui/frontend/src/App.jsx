import React, { useEffect, useState, useRef } from 'react'
import Sidebar from './components/Sidebar'
import Chat from './components/Chat'
import Dataset from './components/Dataset'

export default function App() {
    const [wsUrl, setWsUrl] = useState('/ws')
    const [activePage, setActivePage] = useState('chat')
    
    // Lifted state for Chat
    const [messages, setMessages] = useState([])
    const [status, setStatus] = useState('disconnected')
    const [isProcessing, setIsProcessing] = useState(false)
    const wsRef = useRef(null)

    useEffect(() => {
        // Try to get backend URL from environment or build a sensible default
        const backendUrl = import.meta.env.VITE_BACKEND_URL || window.location.origin

        // Determine WebSocket protocol based on location protocol
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'

        // Build WebSocket URL
        // If backend is same origin, use /ws; otherwise use full URL
        let wsUrlFinal = '/ws'
        if (backendUrl !== window.location.origin) {
            // Backend is on a different host/port  
            const backendHost = new URL(backendUrl).host
            wsUrlFinal = `${proto}//${backendHost}/ws`
        }

        setWsUrl(wsUrlFinal)
    }, [])

    // WebSocket connection management
    useEffect(() => {
        if (!wsUrl || wsUrl === '/ws' && window.location.hostname === 'localhost' && window.location.port === '5173') {
             // Wait for wsUrl to be fully resolved or skip if default in dev mode might be wrong
             // Actually the initial state '/ws' is fine for production but for dev we wait for the effect above
        }

        const ws = new WebSocket(wsUrl)
        wsRef.current = ws

        ws.onopen = () => setStatus('connected')
        ws.onclose = () => setStatus('disconnected')
        ws.onmessage = (ev) => {
            try {
                const data = JSON.parse(ev.data)
                setMessages((m) => [...m, data])
                
                // If we receive a final response (assistant role), stop processing
                if (data.role === 'assistant') {
                    setIsProcessing(false)
                }
            } catch (e) {
                console.error('invalid ws message', e)
            }
        }

        return () => {
            ws.close()
        }
    }, [wsUrl])

    const sendMessage = (content) => {
        if (!content.trim()) return
        const msg = { role: 'user', content }
        setMessages((m) => [...m, msg])
        setIsProcessing(true)
        wsRef.current?.send(JSON.stringify({ content }))
    }

    return (
        <div className="app-root">
            <Sidebar activePage={activePage} setActivePage={setActivePage} />
            <main className="app-main">
                {activePage === 'chat' && (
                    <Chat 
                        messages={messages} 
                        status={status} 
                        onSendMessage={sendMessage}
                        isProcessing={isProcessing}
                    />
                )}
                {activePage === 'dataset' && <Dataset />}
            </main>
        </div>
    )
}
