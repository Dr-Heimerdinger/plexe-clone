import React, { useState } from 'react'
import { testPostgresConnection, savePostgresConnection } from '../../api/client'

export default function PostgreSQLTab() {
    const [connectionForm, setConnectionForm] = useState({
        host: 'localhost',
        port: '5432',
        username: '',
        password: '',
        database: '',
    })
    const [connecting, setConnecting] = useState(false)
    const [connectionStatus, setConnectionStatus] = useState(null)
    const [tables, setTables] = useState([])

    const handleInputChange = (e) => {
        const { name, value } = e.target
        setConnectionForm((prev) => ({
            ...prev,
            [name]: value,
        }))
    }

    const handleTestConnection = async () => {
        setConnecting(true)
        setConnectionStatus(null)

        try {
            const data = await testPostgresConnection(connectionForm)
            setConnectionStatus({
                type: 'success',
                message: 'Connection successful!',
            })
        } catch (error) {
            console.error('Connection error:', error)
            setConnectionStatus({
                type: 'error',
                message: error.message,
            })
        } finally {
            setConnecting(false)
        }
    }

    const handleSaveConnection = async () => {
        setConnecting(true)
        setTables([])

        try {
            const data = await savePostgresConnection(connectionForm)
            setConnectionStatus({
                type: 'success',
                message: 'Connection saved successfully!',
            })
            if (data.tables) {
                setTables(data.tables)
            }
        } catch (error) {
            console.error('Save error:', error)
            setConnectionStatus({
                type: 'error',
                message: error.message,
            })
        } finally {
            setConnecting(false)
        }
    }

    return (
        <div className="postgres-tab">
            <div className="postgres-form">
                <h3>PostgreSQL Connection Settings</h3>

                <div className="form-group">
                    <label htmlFor="host">Host</label>
                    <input
                        id="host"
                        type="text"
                        name="host"
                        value={connectionForm.host}
                        onChange={handleInputChange}
                        placeholder="localhost"
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="port">Port</label>
                    <input
                        id="port"
                        type="text"
                        name="port"
                        value={connectionForm.port}
                        onChange={handleInputChange}
                        placeholder="5432"
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="database">Database</label>
                    <input
                        id="database"
                        type="text"
                        name="database"
                        value={connectionForm.database}
                        onChange={handleInputChange}
                        placeholder="database_name"
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="username">Username</label>
                    <input
                        id="username"
                        type="text"
                        name="username"
                        value={connectionForm.username}
                        onChange={handleInputChange}
                        placeholder="postgres"
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="password">Password</label>
                    <input
                        id="password"
                        type="password"
                        name="password"
                        value={connectionForm.password}
                        onChange={handleInputChange}
                        placeholder="••••••••"
                    />
                </div>

                {connectionStatus && (
                    <div className={`status-message ${connectionStatus.type}`}>
                        {connectionStatus.type === 'success' ? '✓' : '✕'} {connectionStatus.message}
                    </div>
                )}

                <div className="form-actions">
                    <button
                        className="button secondary"
                        onClick={handleTestConnection}
                        disabled={connecting}
                    >
                        {connecting ? 'Testing...' : 'Test Connection'}
                    </button>
                    <button
                        className="button primary"
                        onClick={handleSaveConnection}
                        disabled={connecting}
                    >
                        {connecting ? 'Saving...' : 'Save Connection'}
                    </button>
                </div>
            </div>

            <div className="postgres-info">
                <h4>Connection Information</h4>
                <p>
                    <strong>Current Host:</strong> {connectionForm.host}:{connectionForm.port}
                </p>
                <p>
                    <strong>Database:</strong> {connectionForm.database || 'Not set'}
                </p>
                <p>
                    <strong>Username:</strong> {connectionForm.username || 'Not set'}
                </p>

                {tables.length > 0 && (
                    <div className="table-list">
                        <h4>Available Tables</h4>
                        <ul>
                            {tables.map((table) => (
                                <li key={table}>{table}</li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    )
}
