import React, { useState } from 'react'
import {
    testPostgresConnection,
    executePostgresQuery,
    combineDatasets,
} from '../../api/client'
import QueryResult from './QueryResult'

export default function PostgreSQLTab() {
    const [connectionForm, setConnectionForm] = useState({
        host: 'localhost',
        port: '5432',
        username: '',
        password: '',
        database: '',
    })
    const [connecting, setConnecting] = useState(false)
    const [combining, setCombining] = useState(false)
    const [connectionStatus, setConnectionStatus] = useState(null)
    const [combineStatus, setCombineStatus] = useState(null)
    const [queryResult, setQueryResult] = useState(null)
    const [showCombine, setShowCombine] = useState(false)

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
        setQueryResult(null)
        setShowCombine(false)
        setCombineStatus(null)

        try {
            await testPostgresConnection(connectionForm)
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

    const handleExecute = async () => {
        setConnecting(true)
        setQueryResult(null)
        setConnectionStatus(null)
        setShowCombine(false)
        setCombineStatus(null)

        try {
            const data = await executePostgresQuery(connectionForm)
            setQueryResult(data)
            setConnectionStatus({
                type: 'success',
                message: 'Query executed successfully!',
            })
            setShowCombine(true)
        } catch (error) {
            console.error('Execute error:', error)
            setConnectionStatus({
                type: 'error',
                message: error.message,
            })
        } finally {
            setConnecting(false)
        }
    }

    const handleCombine = async () => {
        setCombining(true)
        setCombineStatus(null)
        try {
            await combineDatasets(
                queryResult.tables,
                queryResult.relationships,
                connectionForm
            )
            setCombineStatus({
                type: 'success',
                message: 'Dataset combination started successfully!',
            })
        } catch (error) {
            console.error('Combine error:', error)
            setCombineStatus({
                type: 'error',
                message: error.message,
            })
        } finally {
            setCombining(false)
        }
    }

    return (
        <div className="postgres-tab">
            <div className="postgres-form">
                <h3>PostgreSQL Connection</h3>
                <p className="form-description">
                    Enter your database credentials to fetch schema information.
                </p>

                <div className="form-grid">
                    <div className="form-group">
                        <label htmlFor="host">Host</label>
                        <input
                            id="host"
                            type="text"
                            name="host"
                            value={connectionForm.host}
                            onChange={handleInputChange}
                            placeholder="e.g., localhost"
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
                            placeholder="e.g., 5432"
                        />
                    </div>
                </div>

                <div className="form-group">
                    <label htmlFor="database">Database</label>
                    <input
                        id="database"
                        type="text"
                        name="database"
                        value={connectionForm.database}
                        onChange={handleInputChange}
                        placeholder="e.g., my_database"
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
                        placeholder="e.g., postgres"
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

                <div className="form-actions">
                    <button
                        className="button secondary"
                        onClick={handleTestConnection}
                        disabled={connecting || combining}
                    >
                        {connecting ? 'Testing...' : 'Test Connection'}
                    </button>
                    <button
                        className="button primary"
                        onClick={handleExecute}
                        disabled={connecting || combining}
                    >
                        {connecting ? 'Executing...' : 'Execute'}
                    </button>
                </div>

                {connectionStatus && (
                    <div className={`status-message ${connectionStatus.type}`}>
                        {connectionStatus.message}
                    </div>
                )}
            </div>

            <div className="postgres-results">
                {queryResult && (
                    <QueryResult
                        tables={queryResult.tables || []}
                        relationships={queryResult.relationships || []}
                    />
                )}
                {showCombine && (
                    <div className="combine-section">
                        <p>
                            Do you want to combine these tables into a final dataset?
                        </p>
                        <button
                            className="button primary"
                            onClick={handleCombine}
                            disabled={combining}
                        >
                            {combining ? 'Combining...' : 'Combine Datasets'}
                        </button>
                        {combineStatus && (
                            <div
                                className={`status-message ${combineStatus.type}`}
                            >
                                {combineStatus.message}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}
