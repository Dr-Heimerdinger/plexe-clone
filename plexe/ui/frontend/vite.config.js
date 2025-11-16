import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            // Proxy /ws to backend during development
            '/ws': {
                target: 'http://localhost:8000',
                ws: true,
                changeOrigin: true,
            },
            // Proxy /api requests if needed in future
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            },
        },
    },
})
