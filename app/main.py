"""
AI Data Analyst Mode - Main Streamlit Application
A powerful tool for analyzing datasets using natural language and AI.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import local modules
from app.config import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_LAYOUT,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    SESSION_KEYS,
    FIGURE_SIZE,
    DPI,
    CHART_COLORS,
)
from app.utils import DataLoader, get_gemini_client, DataAnalyzer


# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
)

# ── Custom styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

/* ── Root token system ── */
:root {
    --bg:              #050b14;
    --bg-surface:      #0d1526;
    --bg-raised:       #141f35;
    --bg-hover:        #1a2844;
    --glass:           rgba(255, 255, 255, 0.03);
    --glass-md:        rgba(255, 255, 255, 0.055);
    --border:          rgba(255, 255, 255, 0.07);
    --border-soft:     rgba(255, 255, 255, 0.04);
    --border-accent:   rgba(124, 58, 237, 0.4);

    --violet:          #7c3aed;
    --violet-bright:   #a78bfa;
    --violet-soft:     rgba(124, 58, 237, 0.10);
    --violet-glow:     rgba(124, 58, 237, 0.25);
    --cyan:            #06b6d4;
    --cyan-bright:     #67e8f9;
    --cyan-soft:       rgba(6, 182, 212, 0.08);
    --rose:            #f43f5e;
    --amber:           #f59e0b;
    --emerald:         #10b981;

    --text-1:          #eef2ff;
    --text-2:          #94a3b8;
    --text-3:          #4a5568;
    --text-4:          #2d3748;

    --shadow-sm:       0 2px 8px rgba(0,0,0,0.35);
    --shadow-md:       0 8px 24px rgba(0,0,0,0.45), 0 2px 8px rgba(0,0,0,0.3);
    --shadow-lg:       0 24px 48px rgba(0,0,0,0.6), 0 8px 16px rgba(0,0,0,0.4);
    --glow-violet:     0 0 16px rgba(124,58,237,0.3), 0 0 48px rgba(124,58,237,0.12);
    --glow-cyan:       0 0 16px rgba(6,182,212,0.25), 0 0 48px rgba(6,182,212,0.08);

    --radius-sm:       8px;
    --radius:          12px;
    --radius-lg:       18px;
    --radius-xl:       24px;
}

/* ── Base resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-1) !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}

/* ── App background — aurora atmosphere ── */
.stApp {
    background-color: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 55% 45% at 8% 92%, rgba(124,58,237,0.14) 0%, transparent 65%),
        radial-gradient(ellipse 45% 35% at 92% 8%,  rgba(6,182,212,0.09) 0%, transparent 65%),
        radial-gradient(ellipse 70% 55% at 50% 110%, rgba(124,58,237,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 60% 60% at 50% 50%,  rgba(13,21,38,0.8) 0%, transparent 100%) !important;
    min-height: 100vh !important;
}

/* ── Main content area ── */
.main .block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1180px !important;
}

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0a1020 0%, #060c1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* Sidebar text fallback */
[data-testid="stSidebar"] * {
    color: var(--text-1) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: var(--text-2) !important;
    font-size: 0.82rem !important;
    line-height: 1.75 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-1) !important;
    letter-spacing: -0.02em !important;
}

/* Sidebar uploader - OVERRIDES ADDED FOR BASEWEB */
[data-testid="stSidebar"] .stFileUploader,
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"],
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: var(--bg-surface) !important;
    background: var(--bg-surface) !important;
    border: 1.5px dashed rgba(255,255,255,0.10) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.25s ease, background 0.25s ease !important;
}
[data-testid="stSidebar"] .stFileUploader:hover,
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(124,58,237,0.4) !important;
    background: var(--violet-soft) !important;
    background-color: var(--violet-soft) !important;
}
[data-testid="stSidebar"] div[data-testid="stUploadedFile"],
[data-testid="stSidebar"] div[data-testid="stUploadedFile"] > div,
[data-testid="stSidebar"] div[data-testid="stUploadedFile"] > div > div {
    background-color: var(--bg-surface) !important;
    background: var(--bg-surface) !important;
    color: var(--text-1) !important;
}
[data-testid="stSidebar"] .stFileUploader * {
    color: var(--text-2) !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: var(--text-3) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar text input - OVERRIDES ADDED FOR BASEWEB */
[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
    background-color: var(--bg-surface) !important;
    background: var(--bg-surface) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: transparent !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
    -webkit-text-fill-color: var(--text-1) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--violet) !important;
    background: var(--violet-soft) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12), var(--glow-violet) !important;
    outline: none !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: var(--text-4) !important;
    -webkit-text-fill-color: var(--text-4) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar hr */
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid var(--border-soft) !important;
    margin: 1rem 0 !important;
}

/* Sidebar alerts */
[data-testid="stSidebar"] .stAlert {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.78rem !important;
}

/* ════════════════════════════════════════
   TYPOGRAPHY
════════════════════════════════════════ */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.05em !important;
    line-height: 1.05 !important;
    background: linear-gradient(125deg, #eef2ff 0%, #a78bfa 45%, #67e8f9 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    color: var(--text-2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-top: 2.25rem !important;
    padding-bottom: 0.6rem !important;
    border-bottom: 1px solid var(--border) !important;
    position: relative !important;
}

h2::after {
    content: '' !important;
    position: absolute !important;
    bottom: -1px !important;
    left: 0 !important;
    width: 40px !important;
    height: 2px !important;
    background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
    border-radius: 2px !important;
}

h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: var(--text-3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
}

/* Paragraphs & lists in main content */
.stMarkdown p {
    font-size: 0.9rem !important;
    line-height: 1.75 !important;
    color: var(--text-2) !important;
}

.stMarkdown strong {
    color: var(--text-1) !important;
    font-weight: 600 !important;
}

/* ════════════════════════════════════════
   METRIC CARDS
════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, var(--bg-raised) 0%, var(--bg-surface) 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem 1.75rem !important;
    box-shadow: var(--shadow-md), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Top-edge gradient activation — the "signature" hover moment */
[data-testid="metric-container"]::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 2px !important;
    background: linear-gradient(90deg, var(--violet), var(--cyan-bright)) !important;
    opacity: 0 !important;
    transition: opacity 0.3s ease !important;
}

/* Ambient glow orb behind metric */
[data-testid="metric-container"]::after {
    content: '' !important;
    position: absolute !important;
    top: -20px !important;
    right: -20px !important;
    width: 80px !important;
    height: 80px !important;
    background: radial-gradient(circle, rgba(124,58,237,0.06) 0%, transparent 70%) !important;
    transition: opacity 0.3s ease !important;
    opacity: 0 !important;
}

[data-testid="metric-container"]:hover {
    border-color: rgba(124,58,237,0.25) !important;
    box-shadow: var(--shadow-lg), var(--glow-violet), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    transform: translateY(-3px) !important;
}

[data-testid="metric-container"]:hover::before,
[data-testid="metric-container"]:hover::after {
    opacity: 1 !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--text-3) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: var(--text-1) !important;
    letter-spacing: -0.04em !important;
    line-height: 1.2 !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    margin-top: 0.15rem !important;
}

/* ════════════════════════════════════════
   BUTTONS
════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--violet) 0%, #5b21b6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.35) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 60%) !important;
    opacity: 0 !important;
    transition: opacity 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.5) !important;
    background: linear-gradient(135deg, #8b5cf6 0%, var(--violet) 100%) !important;
}

.stButton > button:hover::before {
    opacity: 1 !important;
}

.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.3) !important;
}

/* ════════════════════════════════════════
   CHAT MESSAGES
════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.4rem !important;
    margin: 0.5rem 0 !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.2s ease !important;
}

/* User messages get a subtle violet tint */
[data-testid="stChatMessage"][data-role="user"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.07) 0%, var(--bg-surface) 100%) !important;
    border-color: rgba(124,58,237,0.15) !important;
}

/* ════════════════════════════════════════
   CHAT INPUT
════════════════════════════════════════ */
/* Kill Streamlit's massive white bottom bar */
div[data-testid="stBottom"],
div[data-testid="stBottomBlock"],
div[data-testid="stBottom"] > div {
    background-color: transparent !important;
    background: transparent !important;
}

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="textarea"] > div {
    background-color: var(--bg-surface) !important;
    background: var(--bg-surface) !important;
}

[data-testid="stChatInput"] {
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--violet) !important;
    box-shadow: var(--shadow-sm), 0 0 0 3px rgba(124,58,237,0.12), var(--glow-violet) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--text-1) !important;
    -webkit-text-fill-color: var(--text-1) !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-3) !important;
    -webkit-text-fill-color: var(--text-3) !important;
    font-style: italic !important;
}

/* ════════════════════════════════════════
   CODE BLOCKS
════════════════════════════════════════ */
.stCode, pre {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.79rem !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    background: #010409 !important;
    color: #c9d1d9 !important;
    line-height: 1.7 !important;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.3) !important;
}

code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.79rem !important;
    background: rgba(255,255,255,0.06) !important;
    color: var(--violet-bright) !important;
    border-radius: 4px !important;
    padding: 0.15em 0.4em !important;
}

pre code {
    background: transparent !important;
    color: inherit !important;
    padding: 0 !important;
}

/* ════════════════════════════════════════
   DATAFRAMES / TABLES
════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stDataFrame"] th {
    background: var(--bg-raised) !important;
    color: var(--text-3) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 0.7rem 1rem !important;
    border-bottom: 1px solid var(--border) !important;
}

[data-testid="stDataFrame"] td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-2) !important;
    padding: 0.5rem 1rem !important;
    border-bottom: 1px solid var(--border-soft) !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}

[data-testid="stDataFrame"] tr:hover td {
    background: var(--violet-soft) !important;
    color: var(--text-1) !important;
}

/* ════════════════════════════════════════
   EXPANDER
════════════════════════════════════════ */
.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-surface) !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stExpander:hover {
    border-color: rgba(124,58,237,0.2) !important;
}

.stExpander summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: var(--text-2) !important;
    padding: 0.9rem 1.15rem !important;
    background: var(--bg-raised) !important;
    border-bottom: 1px solid var(--border) !important;
    letter-spacing: 0.01em !important;
    transition: color 0.2s ease, background 0.2s ease !important;
}

.stExpander summary:hover {
    color: var(--text-1) !important;
    background: var(--bg-hover) !important;
}

/* ════════════════════════════════════════
   ALERT BANNERS
════════════════════════════════════════ */
.stAlert {
    border-radius: var(--radius-sm) !important;
    border-left-width: 3px !important;
    border-left-style: solid !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
}

.stInfo {
    background: var(--cyan-soft) !important;
    border-left-color: var(--cyan) !important;
    color: var(--text-2) !important;
}

.stWarning {
    background: rgba(245,158,11,0.08) !important;
    border-left-color: var(--amber) !important;
}

.stSuccess {
    background: rgba(16,185,129,0.08) !important;
    border-left-color: var(--emerald) !important;
}

.stError {
    background: rgba(244,63,94,0.08) !important;
    border-left-color: var(--rose) !important;
}

/* ════════════════════════════════════════
   SPINNER
════════════════════════════════════════ */
.stSpinner > div {
    border-top-color: var(--violet) !important;
    border-right-color: var(--cyan) !important;
}

/* ════════════════════════════════════════
   DIVIDERS
════════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ════════════════════════════════════════
   SCROLLBAR
════════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--bg-raised);
    border-radius: 99px;
    transition: background 0.2s ease;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* ════════════════════════════════════════
   TABS (if used)
════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--text-3) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-surface) !important;
    color: var(--text-1) !important;
    border-bottom: 2px solid var(--violet) !important;
}

/* ════════════════════════════════════════
   SELECT / DROPDOWN
════════════════════════════════════════ */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
}

/* ════════════════════════════════════════
   PROGRESS
════════════════════════════════════════ */
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
    border-radius: 99px !important;
}

/* ════════════════════════════════════════
   UTILITY CLASSES
════════════════════════════════════════ */
.tag {
    display: inline-block;
    background: var(--violet-soft);
    color: var(--violet-bright);
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    padding: 0.18rem 0.6rem;
    border-radius: 99px;
    margin-right: 0.35rem;
    border: 1px solid rgba(124,58,237,0.2);
    font-family: 'JetBrains Mono', monospace;
}

.label {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    color: var(--text-3);
    margin-bottom: 0.4rem;
}

.question-chip {
    display: inline-block;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.42rem 1rem;
    font-size: 0.8rem;
    color: var(--text-2);
    margin: 0.22rem;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.question-chip:hover {
    border-color: var(--violet);
    background: var(--violet-soft);
    color: var(--violet-bright);
    box-shadow: var(--glow-violet);
    transform: translateY(-1px);
}

.gradient-bar {
    height: 2px;
    background: linear-gradient(90deg, var(--violet), var(--cyan-bright));
    border-radius: 2px;
}

/* ════════════════════════════════════════
   MATPLOTLIB FIGURES
════════════════════════════════════════ */
[data-testid="stImage"],
.stPlot {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if SESSION_KEYS["dataframe"] not in st.session_state:
        st.session_state[SESSION_KEYS["dataframe"]] = None

    if SESSION_KEYS["chat_history"] not in st.session_state:
        st.session_state[SESSION_KEYS["chat_history"]] = []

    if SESSION_KEYS["analysis_steps"] not in st.session_state:
        st.session_state[SESSION_KEYS["analysis_steps"]] = []

    if SESSION_KEYS["generated_code"] not in st.session_state:
        st.session_state[SESSION_KEYS["generated_code"]] = []

    if SESSION_KEYS["insights"] not in st.session_state:
        st.session_state[SESSION_KEYS["insights"]] = ""


def render_sidebar():
    """Render the sidebar with file upload and settings."""
    with st.sidebar:
        # ── Brand ────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding: 1.75rem 1.4rem 0.25rem;">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
                <div style="
                    width:30px; height:30px;
                    background:linear-gradient(135deg,#7c3aed,#06b6d4);
                    border-radius:8px;
                    display:flex; align-items:center; justify-content:center;
                    box-shadow:0 0 18px rgba(124,58,237,0.45);
                    flex-shrink:0;
                ">
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <rect x="1" y="8" width="3" height="7" rx="1" fill="white" opacity="0.9"/>
                        <rect x="6" y="4" width="3" height="11" rx="1" fill="white" opacity="0.9"/>
                        <rect x="11" y="1" width="3" height="14" rx="1" fill="white" opacity="0.9"/>
                    </svg>
                </div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                                font-weight:800; color:#eef2ff; letter-spacing:-0.03em;
                                line-height:1.1;">
                        DataAnalyst
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem;
                                color:#4a5568; text-transform:uppercase; letter-spacing:0.12em;">
                        AI Mode
                    </div>
                </div>
            </div>
            <div style="height:1px; background:linear-gradient(90deg,rgba(124,58,237,0.45),rgba(6,182,212,0.2),transparent);
                        margin-top:1rem;"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── API key ───────────────────────────────────────────────────────────
        st.markdown(
            '<p class="label" style="padding-left:0.1rem;">Gemini API Key</p>',
            unsafe_allow_html=True,
        )
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            label_visibility="collapsed",
            placeholder="AIza••••••••••••••••••",
            help="Get your free key at aistudio.google.com",
        )

        if api_key:
            st.markdown(
                "<div style='display:flex; align-items:center; gap:0.45rem; "
                "margin-top:0.35rem;'>"
                "<span style='display:inline-block; width:6px; height:6px; "
                "border-radius:50%; background:#10b981; "
                "box-shadow:0 0 8px rgba(16,185,129,0.7);'></span>"
                "<span style='font-family:Inter,sans-serif; font-size:0.72rem; "
                "color:#10b981; font-weight:500;'>Key loaded &amp; ready</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── File upload ───────────────────────────────────────────────────────
        st.markdown(
            '<p class="label" style="padding-left:0.1rem;">Upload Dataset</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=ALLOWED_EXTENSIONS,
            label_visibility="collapsed",
            help=f"Accepted: {', '.join(f'.{e}' for e in ALLOWED_EXTENSIONS)}"
                 f" · Max {MAX_FILE_SIZE_MB} MB",
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")
                return None

            try:
                df, format_type = DataLoader.load_data(uploaded_file)
                st.session_state[SESSION_KEYS["dataframe"]] = df

                st.markdown(
                    f"<div style='"
                    f"background:rgba(124,58,237,0.07); "
                    f"border:1px solid rgba(124,58,237,0.22); "
                    f"border-radius:10px; padding:0.9rem 1.05rem; margin-top:0.6rem;'>"
                    f"<div style='font-family:JetBrains Mono,monospace; font-size:0.74rem; "
                    f"color:#eef2ff; margin-bottom:0.5rem; white-space:nowrap; "
                    f"overflow:hidden; text-overflow:ellipsis;'>"
                    f"📄 {uploaded_file.name}</div>"
                    f"<div style='display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;'>"
                    f"<span style='font-family:JetBrains Mono,monospace; font-size:0.65rem; "
                    f"font-weight:700; background:rgba(124,58,237,0.18); color:#a78bfa; "
                    f"padding:0.15rem 0.55rem; border-radius:99px; letter-spacing:0.06em; "
                    f"border:1px solid rgba(124,58,237,0.25);'>{format_type.upper()}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace; font-size:0.7rem; "
                    f"color:#4a5568;'>{len(df):,} rows · {len(df.columns)} cols</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Could not load file: {str(e)}")
                return None

        st.markdown("---")

        # ── Clear session ─────────────────────────────────────────────────────
        if st.button("↺  Clear Session", use_container_width=True):
            for key in SESSION_KEYS.values():
                if key in st.session_state:
                    st.session_state[key] = None if key == SESSION_KEYS["dataframe"] else (
                        [] if isinstance(st.session_state[key], list) else ""
                    )
            st.rerun()

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="position:absolute; bottom:1.5rem; left:0; right:0;
                    padding:0 1.4rem; text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem;
                        color:#2d3748; letter-spacing:0.08em;">
                powered by gemini · v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

    return api_key


def render_header():
    """Render the main header."""
    st.markdown("""
    <div style="padding: 2.5rem 0 0.75rem;">
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.9rem;">
            <div style="height:1px; width:20px;
                        background:linear-gradient(90deg,transparent,#7c3aed);"></div>
            <span style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700;
                         text-transform:uppercase; letter-spacing:0.14em; color:#7c3aed;">
                AI-Powered Analytics
            </span>
        </div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:2.75rem; font-weight:800;
                    letter-spacing:-0.05em; line-height:1.05;
                    background:linear-gradient(125deg,#eef2ff 0%,#a78bfa 45%,#67e8f9 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; margin:0 0 0.75rem;">
            AI Data Analyst
        </div>
        <div style="height:2px; width:52px;
                    background:linear-gradient(90deg,#7c3aed,#67e8f9);
                    border-radius:2px; margin-bottom:1rem;"></div>
        <p style="font-family:'Inter',sans-serif; font-size:0.9rem; color:#94a3b8;
                  max-width:480px; line-height:1.75; margin:0;">
            Upload any dataset and ask questions in plain English.
            Get instant analysis, visualisations, and AI-powered insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")


def render_dataset_summary(df: pd.DataFrame):
    """Render dataset summary section."""
    st.header("Dataset Summary")

    summary = DataLoader.get_dataset_summary(df)

    # ── Metric strip ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{summary['rows']:,}")
    with col2:
        st.metric("Columns", summary["columns"])
    with col3:
        total_missing = sum(summary["missing_values"].values())
        missing_pct = (
            total_missing / (summary["rows"] * summary["columns"]) * 100
            if summary["rows"] and summary["columns"] else 0
        )
        st.metric(
            "Missing Values",
            f"{total_missing:,}",
            delta=f"{missing_pct:.1f}% of cells",
            delta_color="inverse",
        )
    with col4:
        numeric_count = len(summary["numeric_columns"])
        st.metric(
            "Numeric Columns",
            numeric_count,
            delta=f"{summary['columns'] - numeric_count} categorical",
        )

    st.markdown("---")

    # ── Column info + sample data ─────────────────────────────────────────────
    col_left, col_right = st.columns([3, 4], gap="large")

    with col_left:
        st.subheader("Column Information")

        col_info_df = pd.DataFrame({
            "Column":    summary["column_names"],
            "Type":      [summary["dtypes"][c] for c in summary["column_names"]],
            "Missing":   [summary["missing_values"][c] for c in summary["column_names"]],
            "Missing %": [
                f"{summary['missing_percentages'][c]:.1f}%"
                for c in summary["column_names"]
            ],
        })

        def _highlight_missing(val):
            try:
                pct = float(val.replace("%", ""))
                if pct > 20:
                    return "color: #f43f5e; font-weight:600;"
                if pct > 5:
                    return "color: #f59e0b;"
            except Exception:
                pass
            return ""

        styled = (
            col_info_df.style
            .applymap(_highlight_missing, subset=["Missing %"])
            .set_properties(**{
                "font-family": "JetBrains Mono, monospace",
                "font-size":   "0.78rem",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Sample Data")
        st.dataframe(df.head(5), use_container_width=True)

    # ── Numeric stats ─────────────────────────────────────────────────────────
    if summary["numeric_columns"]:
        st.subheader("Numeric Statistics")
        desc = df[summary["numeric_columns"]].describe().round(4)
        st.dataframe(desc, use_container_width=True)


def render_chat_interface(api_key: str):
    """Render the chat interface for asking questions."""
    st.header("Ask Your Data")

    df = st.session_state[SESSION_KEYS["dataframe"]]

    if df is None:
        st.info("Upload a dataset from the sidebar to get started.")
        return

    # ── Chat history ──────────────────────────────────────────────────────────
    for message in st.session_state[SESSION_KEYS["chat_history"]]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Input ─────────────────────────────────────────────────────────────────
    user_question = st.chat_input(
        "e.g. Why are sales dropping? Which segment has the highest churn?",
        disabled=df is None,
    )

    if user_question and not api_key:
        st.warning("Add your Gemini API key in the sidebar first.")
        return

    if user_question and api_key:
        st.session_state[SESSION_KEYS["chat_history"]].append({
            "role": "user",
            "content": user_question,
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                try:
                    client = get_gemini_client(api_key)

                    summary         = DataLoader.get_dataset_summary(df)
                    dataset_summary = DataLoader.format_summary_for_ai(summary)
                    column_info     = DataLoader.get_column_info(df)

                    # ── Analysis steps ────────────────────────────────────────
                    steps = client.generate_analysis_steps(
                        user_question, dataset_summary, column_info,
                    )
                    st.session_state[SESSION_KEYS["analysis_steps"]] = steps

                    st.markdown("**Analysis Plan**")
                    for i, step in enumerate(steps, 1):
                        st.markdown(
                            f"<div style='display:flex; align-items:flex-start; "
                            f"gap:0.75rem; margin:0.35rem 0; "
                            f"padding:0.65rem 0.9rem; "
                            f"background:rgba(124,58,237,0.05); "
                            f"border:1px solid rgba(124,58,237,0.1); "
                            f"border-radius:8px; transition:background 0.2s ease;'>"
                            f"<span style='font-family:JetBrains Mono,monospace; "
                            f"font-size:0.68rem; font-weight:700; "
                            f"background:linear-gradient(135deg,#7c3aed,#06b6d4); "
                            f"color:white; border-radius:6px; padding:0.12rem 0.5rem; "
                            f"flex-shrink:0; letter-spacing:0.04em;'>{i:02d}</span>"
                            f"<span style='font-family:Inter,sans-serif; font-size:0.86rem; "
                            f"color:#c9d1d9; padding-top:0.05rem; line-height:1.6;'>"
                            f"{step}</span></div>",
                            unsafe_allow_html=True,
                        )

                    # ── Execute steps ─────────────────────────────────────────
                    results  = []
                    analyzer = DataAnalyzer(df)

                    for step in steps:
                        code   = client.generate_analysis_code(
                            step, dataset_summary, column_info, df.shape
                        )
                        result = analyzer.run_analysis_step(code, step)
                        results.append(result)

                        with st.expander(f"↳  {step}", expanded=False):
                            st.code(code, language="python")
                            if result["output"]:
                                st.text(result["output"])

                        for fig in result["figures"]:
                            st.pyplot(fig)

                    # ── Insights ──────────────────────────────────────────────
                    st.markdown("**Insights**")
                    with st.spinner("Generating insights…"):
                        from app.utils.analyzer import format_analysis_results
                        analysis_text = format_analysis_results(results)
                        insights = client.generate_insights(
                            analysis_text, user_question, dataset_summary
                        )

                    st.session_state[SESSION_KEYS["insights"]] = insights

                    st.markdown(
                        f"<div style='"
                        f"background:linear-gradient(135deg,"
                        f"rgba(124,58,237,0.07) 0%,rgba(6,182,212,0.04) 100%); "
                        f"border:1px solid rgba(124,58,237,0.2); "
                        f"border-radius:12px; "
                        f"padding:1.3rem 1.6rem; margin-top:0.85rem; "
                        f"font-family:Inter,sans-serif; font-size:0.9rem; "
                        f"line-height:1.8; color:#c9d1d9; "
                        f"box-shadow:0 0 32px rgba(124,58,237,0.08);'>"
                        f"{insights}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Persist to history
                    st.session_state[SESSION_KEYS["chat_history"]].append({
                        "role": "assistant",
                        "content": (
                            "**Analysis Plan**\n\n"
                            + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
                            + f"\n\n**Insights**\n\n{insights}"
                        ),
                    })

                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
                    st.caption("Check your API key and dataset, then try again.")


def render_sample_questions():
    """Render sample questions for user reference."""
    st.header("Sample Questions")
    st.markdown(
        "<p style='font-family:Inter,sans-serif; font-size:0.86rem; color:#94a3b8; "
        "margin:-0.4rem 0 1.1rem;'>"
        "Try one of these prompts once your dataset is loaded.</p>",
        unsafe_allow_html=True,
    )

    sample_questions = [
        "Why are sales dropping?",
        "What factors affect customer churn?",
        "Which products perform best?",
        "Show sales distribution by region",
        "Compare last year vs this year",
        "Find the top 10 products by revenue",
    ]

    chips_html = "".join(
        f"<span class='question-chip'>🔍&nbsp; {q}</span>" for q in sample_questions
    )
    st.markdown(
        f"<div style='line-height:2.4;'>{chips_html}</div>",
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point."""
    initialize_session_state()

    api_key = render_sidebar()
    render_header()

    df = st.session_state[SESSION_KEYS["dataframe"]]

    if df is not None:
        with st.expander("View Dataset Summary", expanded=False):
            render_dataset_summary(df)
        render_chat_interface(api_key)
    else:
        # ── Empty state ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="
            background:linear-gradient(145deg,#0d1526 0%,#141f35 100%);
            border:1px solid rgba(255,255,255,0.07);
            border-radius:18px;
            padding:3.5rem 2.5rem;
            margin:1.5rem 0;
            text-align:center;
            box-shadow:0 24px 48px rgba(0,0,0,0.5),
                       inset 0 1px 0 rgba(255,255,255,0.04);
            position:relative;
            overflow:hidden;
        ">
            <!-- ambient glow -->
            <div style="
                position:absolute; top:-60px; left:50%; transform:translateX(-50%);
                width:260px; height:260px;
                background:radial-gradient(circle,rgba(124,58,237,0.10) 0%,transparent 70%);
                pointer-events:none;
            "></div>
            <!-- icon -->
            <div style="
                display:inline-flex; align-items:center; justify-content:center;
                width:56px; height:56px;
                background:linear-gradient(135deg,rgba(124,58,237,0.2),rgba(6,182,212,0.15));
                border:1px solid rgba(124,58,237,0.25);
                border-radius:14px;
                font-size:1.6rem;
                margin-bottom:1.2rem;
                box-shadow:0 0 24px rgba(124,58,237,0.2);
            ">📊</div>
            <div style="
                font-family:'Space Grotesk',sans-serif;
                font-size:1.15rem; font-weight:700;
                color:#eef2ff; margin-bottom:0.5rem;
                letter-spacing:-0.02em;
            ">
                No dataset loaded
            </div>
            <p style="
                font-family:'Inter',sans-serif;
                font-size:0.85rem; color:#94a3b8;
                max-width:300px; margin:0 auto; line-height:1.75;
            ">
                Upload a CSV, Excel, or JSON file from the sidebar
                to start asking questions about your data.
            </p>
            <!-- gradient bottom line -->
            <div style="
                position:absolute; bottom:0; left:0; right:0;
                height:2px;
                background:linear-gradient(90deg,transparent,rgba(124,58,237,0.4),rgba(6,182,212,0.3),transparent);
            "></div>
        </div>
        """, unsafe_allow_html=True)

        render_sample_questions()


if __name__ == "__main__":
    main()