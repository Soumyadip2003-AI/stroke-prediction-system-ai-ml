/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'inter': ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        // Brand accent preserved from the original identity.
        accent: '#667eea',
        'accent-deep': '#764ba2',
        // One cool-grey neutral family, harmonised to the indigo accent.
        ink: {
          900: '#0b0d12',
          800: '#12151d',
          700: '#1a1e29',
          600: '#252b3a',
        },
        fog: {
          100: '#eef1f6',
          300: '#c3cad8',
          400: '#98a1b4',
        },
      },
      borderRadius: {
        // One radius scale: pill for interactive, card for surfaces, input for fields.
        'card': '16px',
        'input': '8px',
      },
      transitionTimingFunction: {
        'ease-out-expo': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
    },
  },
  plugins: [],
}
