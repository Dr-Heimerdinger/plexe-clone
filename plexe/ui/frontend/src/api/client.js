/**
 * API Client for Plexe Frontend
 * Handles all HTTP requests to the backend
 */

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

/**
 * Upload file(s) to the backend
 * @param {File[]} files - Array of files to upload
 * @returns {Promise<Object>} Upload response
 */
export async function uploadFiles(files) {
    const formData = new FormData()

    // Add all files to FormData
    files.forEach((file) => {
        formData.append('files', file)
    })

    const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
        headers: {
            // Don't set Content-Type, let browser set it with boundary
        },
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`)
    }

    return await response.json()
}

/**
 * Test PostgreSQL connection
 * @param {Object} connectionConfig - PostgreSQL connection config
 * @returns {Promise<Object>} Test result
 */
export async function testPostgresConnection(connectionConfig) {
    const response = await fetch(`${API_BASE_URL}/api/postgres/test`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(connectionConfig),
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Connection test failed')
    }

    return await response.json()
}

/**
 * Execute a query on a PostgreSQL database
 * @param {Object} connectionConfig - PostgreSQL connection config
 * @returns {Promise<Object>} Query result
 */
export async function executePostgresQuery(connectionConfig) {
    const response = await fetch(`${API_BASE_URL}/api/postgres/execute`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(connectionConfig),
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to execute query')
    }

    return await response.json()
}

/**
 * Get list of uploaded datasets
 * @returns {Promise<Array>} List of datasets
 */
export async function listDatasets() {
    const response = await fetch(`${API_BASE_URL}/api/datasets`, {
        method: 'GET',
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to fetch datasets')
    }

    return await response.json()
}

/**
 * Delete a dataset
 * @param {string} datasetId - ID of dataset to delete
 * @returns {Promise<Object>} Delete result
 */
export async function deleteDataset(datasetId) {
    const response = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}`, {
        method: 'DELETE',
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to delete dataset')
    }

    return await response.json()
}

/**
 * Download a dataset
 * @param {string} datasetId - ID of dataset to download
 * @returns {Promise<Blob>} Downloaded file as a Blob
 */
export async function downloadDataset(datasetId) {
    const response = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}/download`, {
        method: 'GET',
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to download dataset')
    }

    return await response.blob()
}

/**
 * Combine datasets using featuretools
 * @param {Array} tables - List of tables
 * @param {Array} relationships - List of relationships
 * @param {Object} connection - PostgreSQL connection config
 * @returns {Promise<Object>} Combination result
 */
export async function combineDatasets(tables, relationships, connection) {
    const response = await fetch(`${API_BASE_URL}/api/datasets/combine`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tables, relationships, connection }),
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to combine datasets')
    }

    return await response.json()
}
