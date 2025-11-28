import React from 'react';

export default function QueryResult({ tables, relationships }) {
    if (!tables.length && !relationships.length) {
        return null;
    }

    return (
        <div className="query-result-container">
            {tables.length > 0 && (
                <div className="result-section">
                    <h4>Available Tables</h4>
                    <ul className="result-list">
                        {tables.map((table) => (
                            <li key={table} className="result-item table-item">
                                <span className="icon">📄</span>
                                {table}
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {relationships.length > 0 && (
                <div className="result-section">
                    <h4>Table Relationships</h4>
                    <ul className="result-list">
                        {relationships.map((rel, i) => (
                            <li key={i} className="result-item relationship-item">
                                <span className="icon">🔗</span>
                                <span className="table-name">{rel.table_name}</span>
                                <span className="column-name">({rel.column_name})</span>
                                <span className="arrow">→</span>
                                <span className="table-name">{rel.foreign_table_name}</span>
                                <span className="column-name">({rel.foreign_column_name})</span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
