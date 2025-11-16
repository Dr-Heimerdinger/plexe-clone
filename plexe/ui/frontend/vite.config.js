import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

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
                target: 'http://localhost:8000',
                ws: true,
                changeOrigin: true,
            },
            // Proxy /api requests to backend
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            },
        },
    },
})
