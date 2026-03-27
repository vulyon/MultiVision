import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'

// Copy index.html to templates after build
const copyIndexToTemplates = () => ({
  name: 'copy-index-to-templates',
  closeBundle() {
    const srcPath = path.resolve(__dirname, 'dist/index.html')
    const destPath = path.resolve(__dirname, '../templates/index.html')
    if (fs.existsSync(srcPath)) {
      fs.copyFileSync(srcPath, destPath)
      console.log('Copied index.html to templates/')
    }
  },
})

export default defineConfig({
  plugins: [react(), copyIndexToTemplates()],
  base: '/static/',
  build: {
    outDir: '../static',
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})