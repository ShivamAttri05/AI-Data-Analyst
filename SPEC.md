# AI Data Analyst Mode - Specification

## Project Overview
- **Project Name**: AI Data Analyst Mode
- **Type**: Streamlit Web Application
- **Core Functionality**: Allow users to upload datasets and ask analytical questions in natural language, with AI-powered analysis and visualization
- **Target Users**: Business analysts, data scientists, and non-technical users needing quick data insights

## UI/UX Specification

### Layout Structure
- **Sidebar**: Dataset upload, configuration settings
- **Main Area**: 
  - Dataset preview and summary
  - Chat interface for questions
  - Analysis results and visualizations

### Visual Design
- **Color Scheme**: 
  - Primary: Deep Blue (#1E3A5F)
  - Secondary: Teal (#20B2AA)
  - Accent: Coral (#FF6B6B)
  - Background: Light Gray (#F5F7FA)
- **Typography**: 
  - Headers: Roboto Bold
  - Body: Roboto Regular
- **Spacing**: Consistent 16px padding, 8px margins

### Components
1. **File Uploader**: Drag & drop for CSV/Excel files
2. **Dataset Preview Table**: Sortable, scrollable data view
3. **Stats Cards**: Display dataset metrics
4. **Chat Messages**: User questions and AI responses
5. **Code Blocks**: Syntax-highlighted generated Python code
6. **Charts**: Matplotlib/Seaborn visualizations
7. **Insights Panel**: AI-generated insights

## Functionality Specification

### Core Features

#### 1. Dataset Upload
- Accept CSV files (.csv)
- Accept Excel files (.xlsx, .xls)
- Maximum file size: 50MB
- Auto-detect encoding for CSV

#### 2. Dataset Summary
- Display shape (rows × columns)
- List all column names with data types
- Show missing values count and percentage
- Display first 5 sample rows
- Show basic statistics for numeric columns

#### 3. AI Data Analyst Mode
- Send dataset metadata to Gemini API
- Generate investigation steps based on user question
- Create Python code for each analysis step
- Execute generated code and display results

#### 4. Chat Interface
- Natural language question input
- Display conversation history
- Show analysis steps and results
- Support follow-up questions

#### 5. Visualization
- Auto-generate appropriate charts based on data
- Support: line charts, bar charts, scatter plots, histograms, heatmaps
- Display in clean, readable format

#### 6. Suggested Insights
- AI-generated explanations of findings
- Actionable recommendations based on data

## Technical Architecture

### File Structure
```
ai_data_analyst/
├── app/
│   ├── __init__.py
│   ├── main.py              # Main Streamlit app
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Data loading utilities
│   │   ├── gemini_client.py # Gemini API integration
│   │   └── analyzer.py      # Analysis and code generation
│   └── config.py            # Configuration settings
├── requirements.txt
├── README.md
└── .env.example
```

### Dependencies
- streamlit>=1.28.0
- pandas>=2.0.0
- google-generativeai>=0.3.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- openpyxl>=3.1.0
- python-dotenv>=1.0.0

## Acceptance Criteria

1. ✅ User can upload CSV or Excel file
2. ✅ Dataset summary displays correctly with all metrics
3. ✅ User can ask natural language questions
4. ✅ AI generates appropriate analysis steps
5. ✅ Python code is generated and executed
6. ✅ Visualizations are displayed correctly
7. ✅ Chat history is maintained
8. ✅ Insights are provided after analysis
9. ✅ Follow-up questions work seamlessly
10. ✅ Application runs without errors