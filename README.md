# Social Media Sentiment Analysis for Social Impact

## Project Overview

This project demonstrates how data analysis can inform strategies, identify areas of concern, and combat misinformation within the domain of climate change awareness. By analyzing a dataset of 43,943 climate change-related tweets collected between April 27, 2015, and February 21, 2018, the application provides insights into public perception and sentiment regarding climate change.

## Dataset Description

The dataset contains tweets that were independently labeled by 3 reviewers, with only tweets that all reviewers agreed on being included. Each tweet is classified into one of four categories:

- **2 (News)**: Links to factual news about climate change
- **1 (Pro)**: Supports the belief of man-made climate change
- **0 (Neutral)**: Neither supports nor refutes the belief of man-made climate change
- **-1 (Anti)**: Does not believe in man-made climate change

## Features

### 🔍 **Comprehensive Sentiment Analysis**
- Sentiment distribution analysis across all categories
- Theme identification and frequency analysis
- Text pattern analysis (length, word count, common patterns)
- Advanced sentiment analysis using TextBlob

### 📊 **Multiple Visualization Types**
- **Sentiment Distribution Charts**: Pie charts, bar charts, and percentage breakdowns
- **Theme Analysis**: Word clouds, frequency charts, and sentiment-based theme comparison
- **Text Pattern Analysis**: Histograms, box plots, and pattern frequency charts
- **Interactive Dashboard**: Plotly-based interactive visualizations

### 🏗️ **Modular Architecture**
- **`data_loader.py`**: Handles data loading, preprocessing, and validation
- **`sentiment_analyzer.py`**: Performs sentiment analysis and theme identification
- **`visualizer.py`**: Creates comprehensive visualizations
- **`main.py`**: Orchestrates the complete analysis pipeline

### 📈 **Data Processing & Analysis**
- Efficient data structures using Pandas DataFrames and NumPy arrays
- Comprehensive text preprocessing and cleaning
- Exception handling for robust error management
- Logging system for tracking analysis progress

### 📋 **Reporting & Output**
- Text reports with key findings and recommendations
- CSV summaries for further analysis
- High-quality PNG visualizations
- Interactive HTML dashboard

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download the project files
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Static visualizations
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning utilities
- **textblob**: Natural language processing
- **wordcloud**: Word cloud generation
- **plotly**: Interactive visualizations

## Usage

### Quick Start
Run the complete analysis pipeline:
```bash
python main.py
```

### Interactive Mode
Run with interactive menu:
```bash
python main.py --interactive
```
## Output Structure

After running the analysis, the following outputs are generated:

```
outputs/
├── sentiment_distribution.png      # Sentiment distribution charts
├── theme_analysis.png             # Theme analysis visualizations
├── text_patterns.png              # Text pattern analysis charts
├── interactive_dashboard.html     # Interactive Plotly dashboard
├── sentiment_analysis_report.txt  # Human-readable text report
└── sentiment_summary.csv          # CSV summary of key metrics
```

## Analysis Results

### Key Metrics
- **Total Tweets Analyzed**: 43,943
- **Sentiment Distribution**: Breakdown across all four categories
- **Theme Analysis**: Most frequent terms and climate-related vocabulary
- **Text Patterns**: Length distributions, word counts, and common patterns
- **Recommendations**: Actionable insights based on analysis

## Project Structure

```
Project 1/
├── twitter_sentiment_data.csv     # Input dataset
├── data_loader.py                # Data loading and preprocessing
├── sentiment_analyzer.py         # Sentiment analysis engine
├── visualizer.py                 # Visualization creation
├── main.py                       # Main application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── outputs/                      # Generated outputs (created after running)
```

## Technical Implementation

### Data Structures
- **Lists**: For storing analysis results and recommendations
- **Dictionaries**: For mapping sentiment categories and storing analysis results
- **NumPy Arrays**: For numerical computations and statistics
- **Pandas DataFrames**: For efficient data manipulation and analysis

### Exception Handling
- File reading errors (missing files, parsing issues)
- Data validation errors (empty datasets, missing columns)
- Analysis computation errors
- Visualization generation errors

### File Handling
- **Reading**: CSV data loading with error handling
- **Writing**: Multiple output formats (PNG, HTML, TXT, CSV)
- **Validation**: Data integrity checks and preprocessing

## License

This project is created for educational purposes as part of a Python programming course.
