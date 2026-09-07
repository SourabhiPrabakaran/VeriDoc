/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: "#08090A",       // Near-black main workstation background
        panelBg: "#0F1115",      // Sidebar and card background
        panelHeader: "#15181E",  // Sub-header panels
        borderLight: "#1D222B",  // Thin slate-gray borders
        borderActive: "#323A49", // Borders for active elements
        forensicCyan: "#00F0FF", // High-contrast analytical cyan accent
        forensicGreen: "#00E676", // System ready and success green
        forensicRed: "#FF3860",   // Danger and high-risk finding red
        forensicAmber: "#FFB300", // Warning yellow
        textMain: "#F1F5F9",     // Primary slate white text
        textMuted: "#8A99AD",    // Subtitle and metadata text
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "Liberation Mono", "Courier New", "monospace"],
      }
    },
  },
  plugins: [],
}
