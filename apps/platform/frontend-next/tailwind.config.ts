import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#f4f6fb',
        panel: '#ffffff',
        ink: '#1f2937',
        muted: '#6b7280',
        line: '#e9edf3',
        accent: '#2f6bff',
        accentSoft: '#eef3ff',
        caution: '#f59e0b',
        cautionSoft: '#fff6e6',
        success: '#16a34a',
        successSoft: '#eafaf0',
        danger: '#ef4444',
        dangerSoft: '#feecec',
      },
      boxShadow: {
        panel: '0 8px 24px rgba(15, 23, 42, 0.05)',
      },
      fontFamily: {
        sans: ['PingFang SC', 'Microsoft YaHei', '-apple-system', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SF Mono', 'Menlo', 'monospace'],
      },
      borderRadius: {
        xl2: '1.25rem',
      },
      backgroundImage: {
        grain: 'radial-gradient(circle at 1px 1px, rgba(84,64,24,0.08) 1px, transparent 0)',
      },
    },
  },
  plugins: [],
} satisfies Config
