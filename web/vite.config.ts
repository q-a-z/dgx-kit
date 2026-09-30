import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the page runs here and talks to the DGX-kit service
// (on this machine or a Spark) through this proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': process.env.DGXKIT_BACKEND ?? 'http://localhost:3000' },
  },
})
