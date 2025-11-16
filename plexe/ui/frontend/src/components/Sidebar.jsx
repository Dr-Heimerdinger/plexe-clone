import React from 'react'

export default function Sidebar({ activePage, setActivePage }) {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <h2>Plexe</h2>
                <p>Assistant</p>
            </div>

            <nav className="sidebar-nav">
                <button
                    className={`nav-item ${activePage === 'chat' ? 'active' : ''}`}
                    onClick={() => setActivePage('chat')}
                >
                    <span className="icon">💬</span>
                    <span className="label">Chat</span>
                </button>

                <button
                    className={`nav-item ${activePage === 'dataset' ? 'active' : ''}`}
                    onClick={() => setActivePage('dataset')}
                >
                    <span className="icon">📊</span>
                    <span className="label">Dataset</span>
                </button>
            </nav>

            <div className="sidebar-footer">
                <p className="version">v0.26.2</p>
            </div>
        </aside>
    )
}
