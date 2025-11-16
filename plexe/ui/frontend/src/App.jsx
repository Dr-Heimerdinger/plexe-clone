import React, { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import Chat from './components/Chat'
import Dataset from './components/Dataset'

export default function App() {
    const [wsUrl, setWsUrl] = useState('/ws')
    const [activePage, setActivePage] = useState('chat')

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
            <Sidebar activePage={activePage} setActivePage={setActivePage} />
            <main className="app-main">
                {activePage === 'chat' && <Chat wsUrl={wsUrl} />}
                {activePage === 'dataset' && <Dataset />}
            </main>
        </div>
    )
}
