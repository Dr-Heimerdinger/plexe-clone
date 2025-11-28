import React, { useState, useEffect } from 'react';
import { listDatasets, deleteDataset, downloadDataset } from '../../api/client'; // Corrected import

export default function OverviewTab() {
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchDatasets();
    }, []);

    const fetchDatasets = async () => {
        try {
            setLoading(true);
            const response = await listDatasets(); // Changed to listDatasets
            setDatasets(response.datasets); // Assuming the API returns data in a 'datasets' field
        } catch (err) {
            setError('Failed to fetch datasets.');
            console.error('Error fetching datasets:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (datasetId) => {
        if (window.confirm('Are you sure you want to delete this dataset?')) {
            try {
                await deleteDataset(datasetId);
                fetchDatasets(); // Refresh the list
            } catch (err) {
                setError('Failed to delete dataset.');
                console.error('Error deleting dataset:', err);
            }
        }
    };

    const handleDownload = async (datasetId, datasetName) => {
        try {
            const response = await downloadDataset(datasetId); // downloadDataset returns a blob
            const url = window.URL.createObjectURL(response); // Create URL from blob
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', datasetName); // Or use a name from the response
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url); // Clean up the URL object
        } catch (err) {
            setError('Failed to download dataset.');
            console.error('Error downloading dataset:', err);
        }
    };

    if (loading) {
        return <div className="overview-tab-container">Loading datasets...</div>;
    }

    if (error) {
        return <div className="overview-tab-container error">{error}</div>;
    }

    return (
        <div className="overview-tab-container">
            <h3>Available Datasets</h3>
            {datasets.length === 0 ? (
                <p>No datasets available yet. Upload or connect one!</p>
            ) : (
                <div className="dataset-cards-grid">
                    {datasets.map((dataset) => (
                        <div key={dataset.id} className="dataset-card">
                            <h4>{dataset.filename}</h4> {/* Display filename */}
                            <p><strong>Created:</strong> {new Date(dataset.created_at * 1000).toLocaleDateString()}</p> {/* Convert timestamp to date */}
                            <p><strong>Size:</strong> {(dataset.size / 1024).toFixed(2)} KB</p> {/* Convert bytes to KB */}
                            <div className="card-actions">
                                <button onClick={() => handleDownload(dataset.id, dataset.filename)}>Download</button>
                                <button onClick={() => handleDelete(dataset.id)} className="delete-button">Delete</button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
