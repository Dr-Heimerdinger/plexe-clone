import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Use environment variable for backend URL, defaulting to Docker service name
// When running in Docker, use 'backend:8100'. When running locally, use 'localhost:8100'
const backendHost = process.env.VITE_BACKEND_HOST || 'backend'
const backendUrl = `http://${backendHost}:8100`

export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0',
        port: 3000,
        strictPort: false,
        hmr: {
            host: 'localhost',
            port: 3000,
            protocol: 'ws',
        },
        proxy: {
            // Proxy /ws to backend during development
            '/ws': {
                target: backendUrl,
                ws: true,
                changeOrigin: true,
            },
            // Proxy /api requests to backend
            '/api': {
                target: backendUrl,
                changeOrigin: true,
            },
            // Proxy /health to backend
            '/health': {
                target: backendUrl,
                changeOrigin: true,
            },
        },
    },
})
