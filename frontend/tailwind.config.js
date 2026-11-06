/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        // Surfaces — darkest to lightest
        canvas: '#080b11',
        panel: '#0c1017',
        surface: {
          DEFAULT: '#10151d',
          2: '#151b25',
          3: '#1b222e',
        },
        line: {
          DEFAULT: '#1d2430',
          strong: '#29323f',
        },
        // Text
        ink: {
          DEFAULT: '#e7ebf1',
          muted: '#a0a9b8',
          faint: '#6b7585',
          ghost: '#465061',
        },
        // Accent + semantic
        brand: {
          DEFAULT: '#5b8cff',
          strong: '#4373f0',
        },
        up: '#22c38e',
        down: '#f0525a',
        warn: '#f4b03e',
        info: '#4cb8ff',
        violet: '#a78bfa',
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }], // 11px
      },
      boxShadow: {
        pop: '0 12px 32px -8px rgb(0 0 0 / 0.6), 0 0 0 1px rgb(255 255 255 / 0.04)',
      },
    },
  },
  plugins: [],
};
