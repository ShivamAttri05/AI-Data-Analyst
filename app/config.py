"""
Configuration settings for AI Data Analyst Mode application.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Streamlit configuration
STREAMLIT_PAGE_TITLE = "AI Data Analyst Mode"
STREAMLIT_PAGE_ICON = "📊"
STREAMLIT_LAYOUT = "wide"

# File upload settings
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = ["csv", "xlsx", "xls"]

# Gemini API settings
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_TOKENS = 2048

# Analysis settings
MAX_SAMPLE_ROWS = 5
DEFAULT_TOP_N = 10

# Visualization settings
FIGURE_SIZE = (10, 6)
DPI = 100
STYLE = "seaborn-v0_8-darkgrid"

# Colors for charts
CHART_COLORS = [
    "#1E3A5F",  # Deep Blue
    "#20B2AA",  # Teal
    "#FF6B6B",  # Coral
    "#4ECDC4",  # Mint
    "#45B7D1",  # Sky Blue
    "#96CEB4",  # Sage
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
]

# Session state keys
SESSION_KEYS = {
    "dataset": "dataset",
    "dataframe": "dataframe",
    "chat_history": "chat_history",
    "analysis_steps": "analysis_steps",
    "generated_code": "generated_code",
    "insights": "insights",
}

