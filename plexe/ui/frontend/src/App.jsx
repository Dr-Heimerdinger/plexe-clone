import React, { useEffect, useState } from 'react'
import Chat from './components/Chat'

export default function App() {
    const [wsUrl, setWsUrl] = useState('/ws')

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


    return (
        <div className="app-root">
            <header className="app-header">
                <h1>Plexe Assistant</h1>
                <p className="subtitle">Build ML models through natural conversation</p>
            </header>
            <main className="app-main">
                <Chat wsUrl={wsUrl} />
                {console.log("WebSocket URL:", wsUrl)}
            </main>
        </div>
    )
}
