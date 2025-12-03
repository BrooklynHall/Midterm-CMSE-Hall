# Swimming Performance Analytics and Prediction App

## Project Overview
https://midterm-cmse-hall.streamlit.app

This Streamlit-based web application analyzes historical swimming performance data from Olympic results (1912-2020) and world records to show trends in athlete times, gender equity, event-specific improvements, and the impacts of technological advancements and regulations. The app integrates data from two sources, applies data cleaning and preprocessing, provides interactive visualizations, statistical insights, and machine learning predictions for future performance trends.

The dataset selection was driven by my personal background: 13 years of my own competitive swimming provided deep insight into the sport's dynamics, allowing for analysis of trends like time improvements, gender gaps (women's progress under figures like Katie Ledecky), and the effects of the 2009 tech suit ban. The Olympic dataset (from Kaggle) and world records dataset give exploration of how the sport has evolved, with gradual time reductions over time, plateauing post-ban due to regulatory changes and technique shifts.

Insights from exploratory data analysis (EDA) include:
- **Time Trends**: Swim times have declined steadily over decades, with a notable slowdown after the 2009 tech suit ban, highlighting regulatory impact.
- **Gender Equity**: Women have closed the performance gap with men, though disparities persist in some events.
- **Event Variations**: Shorter events (sprints) show faster improvement due to technique differences, while longer distances rely more on endurance.
- **Outliers and Anomalies**: Removed extreme values from early Olympics (1920s) based on IQR to ensure clean analysis.
- **Data Quality**: Missing patterns are Missing At Random (MAR), correlated with year, allowing imputation.

The app has a tabbed interface with sidebar filters (year, events, sex, source) affecting all visualizations. It has six visualizations, statistical tests, and ML models (Linear Regression, Random Forest, Gradient Boosting).

## Features

### Data Integration & Preparation
- **Two Data Sources**: Olympic swimming results (5,413 rows, 1912-2020) and world records (1,687 rows).
- **Advanced Cleaning**:
  - Standardized formats (Sex: 'M'/'W' → 'Men'/'Women', Stroke mappings, Event naming).
  - Time parsing (MM:SS.SS → seconds for consistency).
  - Imputation: Tested mean, median, forward-fill on simulated 10% missing data; forward-fill selected for minimal bias.
  - Outlier removal: IQR-based (remove values outside Q1-1.5*IQR to Q3+1.5*IQR).
  - Deduplication, null handling, and country decoding (ISO3 to full names).
- **Feature Engineering**: Standardization (Z-score), label encoding for categorical features, polynomial features (Year²) for trend modeling.
- **Preprocessing Scripts**: Integrated into the app; exports cleaned CSVs.

### Visualizations & Analysis
- **Six Interactive Visualizations** (Plotly-based):
  - Histogram: Time distributions.
  - Box Plot: Gender/sex variances.
  - Scatter: Year vs. Time trends with size coloring.
  - Heatmap: Feature correlations.
  - 3D Scatter: Gender gaps over time with z-axis for equity analysis.
  - Trend Line: Event-specific time progressions.
- **Statistical Summaries**:
  - Descriptive stats (mean, std, min, max by source/year).
  - Correlation analysis (Year-Time: strong negative correlation).
  - T-tests for gender differences.
- **Real-World Insights**:
  - Tech suit ban evaluation: Pre-2010 vs. Post-2010 time changes.
  - Equity tracking: Women's catch-up in events like 400m Freestyle.
  - Performance plateau post-regulation.

### Machine Learning & Predictions
- **Models**: Linear Regression, Random Forest (tuned with GridSearchCV), and Gradient Boosting.
- **Training & Evaluation**: 80/20 train/test split; metrics: MAE, RMSE, R², CV Score (MAE).
- **Predictions**: Interactive tool for year/sex/stroke/source; generates ensemble predictions with history tracking.
- **Insights**: RF outperforms for robustness; captures trends into 2030.

### Streamlit App Interface
- **Tabbed Layout**:
  1. Introduction & Data Preparation: Overview, sources, cleaning pipeline.
  2. Data Preparation Process: Step-by-step visualizations (heatmaps, IQR box plots for outliers).
  3. Exploratory Data Analysis: Interactive viz with sidebar filters.
  4. Statistical Summaries & Insights: Stats, tests, and trend analyses.
  5. Predictive Modeling: Model results, predictions, history.
  6. Documentation & Usage: Repo, data dict, rubric alignment.
- **Interactive Elements**: 5+ filters/sliders (year start, event multiselect, sex selectbox, source, viz type, prediction sliders).
- **Caching & Performance**: Handles ~5,000+ rows efficiently with session state for history persistence.
- **Deployment Ready**: Caches model training; supports Streamlit Cloud/GitHub.

## Technologies Used
- **Languages/Python Libraries**: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Plotly, pycountry, Joblib).
- **App Framework**: Streamlit for web interface.
- **Data Sources**: Kaggle (Olympic data), GitHub (World Records data).

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required libraries and versions (see `requirements.txt`):
  streamlit
  pandas
  numpy
  plotly
  scikit-learn
  scipy
  joblib
  pycountry
  matplotlib
  seaborn

### Installation
1. Clone the repository:

   git clone https://github.com/BrooklynHall/final-cmse-hall
   cd final-cmse-hall

2. Install dependencies:

   pip install -r requirements.txt


### File Structure

/
├── data/
│   ├── olympic_swimming_1912_2020.csv  # Raw Olympic data
│   ├── swim_records.csv                # Raw records data
│   ├── cleaned_olympic_data.csv        # Processed Olympic data (exported by app)
│   └── cleaned_records_data.csv        # Processed records data (exported by app)
├── app.py                              # Main Streamlit app
├── requirements.txt                    # Python dependencies
└── README.md                           # This file


## Author & Acknowledgments
- **Author**: Blake Hall (CMSE 830 Final Project).
- Repository: [https://github.com/BrooklynHall/midterm-cmse-hall](https://github.com/BrooklynHall/midterm-cmse-hall).
- Data Sources: Kaggle Olympic dataset, GitHub Swim Records.