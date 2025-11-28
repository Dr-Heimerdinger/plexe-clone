import React, { useState } from 'react'
import UploadTab from './dataset/UploadTab'
import PostgreSQLTab from './dataset/PostgreSQLTab'
import OverviewTab from './dataset/OverviewTab'

export default function Dataset() {
    const [activeTab, setActiveTab] = useState('upload')

    return (
        <div className="dataset-container">
            <div className="dataset-header">
                <h2>Dataset Management</h2>
                <p>Upload or connect to your data sources</p>
            </div>

            <div className="tabs">
                <div className="tab-buttons">
                    <button
                        className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
                        onClick={() => setActiveTab('overview')}
                    >
                        <span className="icon">📊</span>
                        Overview
                    </button>
                    <button
                        className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
                        onClick={() => setActiveTab('upload')}
                    >
                        <span className="icon">📤</span>
                        Upload Data
                    </button>
                    <button
                        className={`tab-button ${activeTab === 'postgres' ? 'active' : ''}`}
                        onClick={() => setActiveTab('postgres')}
                    >
                        <span className="icon">🗄️</span>
                        PostgreSQL
                    </button>

                </div>

                <div className="tab-content">

                    {activeTab === 'upload' && <UploadTab />}
                    {activeTab === 'postgres' && <PostgreSQLTab />}
                    {activeTab === 'overview' && <OverviewTab />}
                </div>
            </div>
        </div>
    )
}
