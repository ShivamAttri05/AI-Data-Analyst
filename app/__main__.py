"""
Entry point for running the AI Data Analyst Mode application.
Run with: streamlit run app/__main__.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import main

if __name__ == "__main__":
    main()

