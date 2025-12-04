# Final Project: Swimming Performance Analytics and Prediction
# Blake Hall - CMSE 830 Final Project
# Streamlit application that integrates two data sources (Olympic Results and World Records) to analyze swimming performance trends, with advanced cleaning, viz, stats, ML, with real-world applications.
# Repos: https://github.com/BrooklynHall/midterm-cmse-hall 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind, pearsonr
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry

st.set_page_config(layout="wide")

if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []
if 'latest_prediction' not in st.session_state:
    st.session_state['latest_prediction'] = None

st.session_state['prediction_history'] = [p for p in st.session_state['prediction_history'] if isinstance(p, dict)]

@st.cache_data
def load_and_prepare_data():
    olympic_df_original = pd.read_csv('data/olympic_swimming_1912_2020.csv')
    olympic_df = olympic_df_original.copy()
    olympic_df.rename(columns={'Team': 'Nationality', 'Results': 'Time', 'Gender': 'Sex', 'Distance (in meters)': 'Distance'}, inplace=True)
    sex_map = {'M': 'Men', 'W': 'Women'}
    olympic_df['Sex'] = olympic_df['Sex'].map(sex_map).fillna(olympic_df['Sex'])
    olympic_df['Distance'] = olympic_df['Distance'].astype(str).str.replace('m', '') + 'm'
    olympic_df['Event'] = olympic_df['Distance'] + ' ' + olympic_df['Stroke']
    olympic_df['Source'] = 'Olympic'

    records_df_original = pd.read_csv('data/swim_records.csv', encoding='cp1252')
    records_df = records_df_original.copy()
    stroke_map = {
        'Free': 'Freestyle',
        'Back': 'Backstroke',
        'Breast': 'Breaststroke',
        'Fly': 'Butterfly',
        'IM': 'Individual Medley'
    }
    records_df['Stroke'] = records_df['Stroke'].map(stroke_map).fillna(records_df['Stroke'])
    records_df['Sex'] = records_df['Sex'].map(sex_map).fillna(records_df['Sex'])
    records_df['Source'] = 'World Records'

    orig_oly = olympic_df.shape[0]
    orig_rec = records_df.shape[0]

    missing_oly_total = olympic_df.isnull().sum().sum()
    missing_rec_total = records_df.isnull().sum().sum()

    figs_heat = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(olympic_df_original.isnull(), cmap="viridis", ax=ax1, cbar=False)
    ax1.set_title("Missing Data - Olympic")
    sns.heatmap(records_df_original.isnull(), cmap="viridis", ax=ax2, cbar=False)
    ax2.set_title("Missing Data - Records")
    figs_heat.append(fig)

    oly_summary = olympic_df.describe()
    rec_summary = records_df.describe()

    histogram_fig = olympic_df['Year'].hist(bins=20, figsize=(10,5))
    plt.title("Year Distribution - Olympic")
    plt.xlabel("Year")
    plt.ylabel("Count")
    histogram_fig.figure

    olympic_df['missing_raw_time'] = olympic_df['Time'].isnull()
    missing_by_year_pct = olympic_df.groupby('Year')['missing_raw_time'].mean()
    years_missing = olympic_df[olympic_df['missing_raw_time']]['Year'].mean()
    years_full = olympic_df[~olympic_df['missing_raw_time']]['Year'].mean()

    if olympic_df.shape[0] > 0:
        time_series = olympic_df['Time'].copy()
        def quick_parse(s):
            try:
                if ':' not in str(s): return float(s)
                parts = str(s).split(':')
                if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
                return np.nan
            except: return np.nan
        time_col = time_series.apply(quick_parse)
        orig_sd = time_col.std() if time_col.std() > 0 else 1
        time_test = time_col.copy()
        time_test.iloc[::10] = np.nan
        def impute_mean(col): return col.fillna(col.mean())
        def impute_median(col): return col.fillna(col.median())
        def impute_ffill(col): return col.fillna(method='ffill')
        mean_imp = impute_mean(time_test.copy())
        median_imp = impute_median(time_test.copy())
        ffill_imp = impute_ffill(time_test.copy())
        mean_sd = mean_imp.std()
        median_sd = median_imp.std()
        ffill_sd = ffill_imp.std()
    else:
        orig_sd = mean_sd = median_sd = ffill_sd = 1

    cleaned_oly = olympic_df.copy()
    cleaned_oly = cleaned_oly[['Year', 'Distance', 'Stroke', 'Sex', 'Nationality', 'Athlete', 'Time', 'Rank', 'Event', 'Source']]
    dupe_drop_oly = cleaned_oly.drop_duplicates().shape[0]
    cleaned_oly = cleaned_oly[cleaned_oly['Time'].notnull()]
    time_clean_oly = cleaned_oly.shape[0]

    def time_seconds(s):
        try:
            s_str = str(s).strip()
            if ':' not in s_str:
                return float(s_str)
            parts = s_str.split(':')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].replace('.', '').isdigit():
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3 and all(p.replace('.', '').isdigit() for p in parts):
                return (float(parts[0]) * 3600) + (float(parts[1]) * 60) + float(parts[2])
            else:
                return np.nan
        except:
            return np.nan

    cleaned_oly['Time_sec'] = cleaned_oly['Time'].apply(time_seconds)
    bad_times = cleaned_oly[cleaned_oly['Time_sec'].isna()]['Time'].unique()
    cleaned_oly = cleaned_oly[cleaned_oly['Time_sec'].notnull()]
    sec_clean_oly = cleaned_oly.shape[0]
    cleaned_oly['Time_sec'] = impute_ffill(cleaned_oly['Time_sec'])
    cleaned_oly['Time_sec'] = cleaned_oly['Time_sec'].fillna(cleaned_oly['Time_sec'].mean())

    def decode(c):
        try:
            return pycountry.countries.get(alpha_3=c).name
        except:
            return c
    cleaned_oly['Nationality_full'] = cleaned_oly['Nationality'].apply(decode)

    cleaned_rec = records_df.copy()
    cleaned_rec['Year'] = cleaned_rec['Date'].str.extract(r'(\d{4})', expand=False).astype(int)
    if 'Athlete' not in cleaned_rec.columns:
        cleaned_rec['Athlete'] = cleaned_rec['Swimmer']
    cleaned_rec = cleaned_rec[['Year', 'Distance', 'Stroke', 'Sex', 'Nationality', 'Athlete', 'Time', 'Event', 'Source']]
    dupe_drop_rec = cleaned_rec.drop_duplicates().shape[0]
    cleaned_rec = cleaned_rec[cleaned_rec['Time'].notnull()]
    time_clean_rec = cleaned_rec.shape[0]
    cleaned_rec['Time_sec'] = cleaned_rec['Time'].apply(time_seconds)
    cleaned_rec = cleaned_rec[cleaned_rec['Time_sec'].notnull()]
    sec_clean_rec = cleaned_rec.shape[0]
    cleaned_rec['Time_sec'] = impute_ffill(cleaned_rec['Time_sec'])
    cleaned_rec['Time_sec'] = cleaned_rec['Time_sec'].fillna(cleaned_rec['Time_sec'].mean())
    cleaned_rec['Distance'] = cleaned_rec['Distance'].astype(str) + 'm'
    cleaned_rec['Event'] = cleaned_rec.apply(lambda row: f"{row['Distance']} {row['Stroke']}", axis=1)
    cleaned_rec['Nationality_full'] = cleaned_rec['Nationality'].apply(decode)

    before_out_oly = cleaned_oly.shape[0]
    cleaned_oly = remove_outliers(cleaned_oly, 'Time_sec')
    after_out_oly = cleaned_oly.shape[0]

    before_out_rec = cleaned_rec.shape[0]
    cleaned_rec = remove_outliers(cleaned_rec, 'Time_sec')
    after_out_rec = cleaned_rec.shape[0]

    return (orig_oly, orig_rec, before_out_oly, after_out_oly, before_out_rec, after_out_rec,
            dupe_drop_oly, time_clean_oly, sec_clean_oly, dupe_drop_rec, time_clean_rec, sec_clean_rec,
            missing_oly_total, missing_rec_total, figs_heat, oly_summary, rec_summary, histogram_fig, 
            missing_by_year_pct, years_missing, years_full, orig_sd, mean_sd, median_sd, ffill_sd, bad_times,
            cleaned_oly.shape[0], cleaned_rec.shape[0], olympic_df_original, records_df_original)

olympic_df = pd.read_csv('data/cleaned_olympic_data.csv')
records_df = pd.read_csv('data/cleaned_records_data.csv')

def remove_outliers(df, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

olympic_df = remove_outliers(olympic_df, 'Time_sec')
records_df = remove_outliers(records_df, 'Time_sec')

scaler_time = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)
le_sex = LabelEncoder()
le_stroke = LabelEncoder()

all_sex = sorted(list(set(olympic_df['Sex'].unique()) | set(records_df['Sex'].unique())))
le_sex.fit(all_sex)
all_stroke = sorted(list(set(olympic_df['Stroke'].unique()) | set(records_df['Stroke'].unique())))
le_stroke.fit(all_stroke)

olympic_df['Time_sec_scaled'] = scaler_time.fit_transform(olympic_df[['Time_sec']].copy())
olympic_df['Sex_encoded'] = le_sex.transform(olympic_df['Sex'])
olympic_df['Stroke_encoded'] = le_stroke.transform(olympic_df['Stroke'])
year_poly_features = poly.fit_transform(olympic_df[['Year']].copy())
olympic_df['Year_poly'] = year_poly_features[:, 1]

records_df['Time_sec_scaled'] = scaler_time.fit_transform(records_df[['Time_sec']].copy())
records_df['Sex_encoded'] = le_sex.transform(records_df['Sex'])
records_df['Stroke_encoded'] = le_stroke.transform(records_df['Stroke'])
year_poly_features_rec = poly.fit_transform(records_df[['Year']].copy())
records_df['Year_poly'] = year_poly_features_rec[:, 1]

@st.cache_data
def train_models(df):
    X = df[['Year', 'Year_poly', 'Sex_encoded', 'Stroke_encoded']]
    y = df['Time_sec']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lin_reg = LinearRegression()
    rf_reg = RandomForestRegressor(random_state=42)
    gb_reg = GradientBoostingRegressor(random_state=42)
    
    rf_params = {'n_estimators': [100, 200]}
    gs_rf = GridSearchCV(rf_reg, rf_params, cv=5, scoring='neg_mean_absolute_error')
    gs_rf.fit(X_train, y_train)
    best_rf = gs_rf.best_estimator_
    
    lin_reg.fit(X_train, y_train)
    gb_reg.fit(X_train, y_train)
    
    y_pred_lin = lin_reg.predict(X_test)
    y_pred_rf = best_rf.predict(X_test)
    y_pred_gb = gb_reg.predict(X_test)
    
    results = {
        'Linear Regression': {
            'MAE': mean_absolute_error(y_test, y_pred_lin),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lin)),
            'R2': r2_score(y_test, y_pred_lin),
            'CV Score (MAE)': -cross_val_score(lin_reg, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
        },
        'Random Forest': {
            'MAE': mean_absolute_error(y_test, y_pred_rf),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'R2': r2_score(y_test, y_pred_rf),
            'CV Score (MAE)': -gs_rf.best_score_
        },
        'Gradient Boosting': {
            'MAE': mean_absolute_error(y_test, y_pred_gb),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
            'R2': r2_score(y_test, y_pred_gb),
            'CV Score (MAE)': -cross_val_score(gb_reg, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
        }
    }
    return results, lin_reg, best_rf, gb_reg

results_oly, lin_reg_oly, rf_reg_oly, gb_reg_oly = train_models(olympic_df)
results_rec, lin_reg_rec, rf_reg_rec, gb_reg_rec = train_models(records_df)

st.title("Swimming Performance Analytics and Prediction App")
st.write("A century of swimming data with advanced visualizations, statistical insights, machine learning predictions, and real-world impacts on performance trends, equity, and technology advancements.")

st.sidebar.header("Filters & Tools")
year_start = st.sidebar.slider("Start Year", int(olympic_df['Year'].min()), int(olympic_df['Year'].max()), int(olympic_df['Year'].min()), key="year_filter")
event_filter = st.sidebar.multiselect("Events", sorted(olympic_df['Event'].unique()), [], key="event_filter")
sex_filter = st.sidebar.selectbox("Sex", ['All', 'Men', 'Women'], index=0, key="sex_filter")
source_filter = st.sidebar.selectbox("Source", ['All', 'Olympic', 'Records'], index=0, key="source_filter")
viz_type = st.sidebar.selectbox("Visualization Type", ['Histogram', 'Box Plot', 'Scatter', 'Heatmap', '3D Scatter', 'Trend Line'], index=0, key="viz_select")

st.sidebar.subheader("Prediction Tool")
pred_year = st.sidebar.slider("Prediction Year", int(olympic_df['Year'].min()), 2030, int(olympic_df['Year'].max()), key="pred_year")
pred_sex = st.sidebar.selectbox("Prediction Sex", ['Men', 'Women'], key="pred_sex")
pred_stroke = st.sidebar.selectbox("Prediction Stroke", sorted(olympic_df['Stroke'].unique()), key="pred_stroke")
pred_source = st.sidebar.selectbox("Prediction Data Source", ['Olympic', 'Records'], key="pred_source")
if st.sidebar.button("Generate Prediction", key="predict_button"):
    sex_enc = 0 if pred_sex == 'Men' else 1
    stroke_enc = le_stroke.transform([pred_stroke])[0]
    year_poly_pred = pred_year ** 2
    X_pred = pd.DataFrame([[pred_year, year_poly_pred, sex_enc, stroke_enc]], columns=['Year', 'Year_poly', 'Sex_encoded', 'Stroke_encoded'])
    
    if pred_source == 'Olympic':
        lin_pred = lin_reg_oly.predict(X_pred)[0]
        rf_pred = rf_reg_oly.predict(X_pred)[0]
        gb_pred = gb_reg_oly.predict(X_pred)[0]
    else:
        lin_pred = lin_reg_rec.predict(X_pred)[0]
        rf_pred = rf_reg_rec.predict(X_pred)[0]
        gb_pred = gb_reg_rec.predict(X_pred)[0]
    
    pred_dict = {'Year': pred_year, 'Sex': pred_sex, 'Stroke': pred_stroke, 'Source': pred_source, 'LinReg': lin_pred, 'RF': rf_pred, 'GB': gb_pred}
    st.session_state['prediction_history'].append(pred_dict)
    st.session_state['latest_prediction'] = pred_dict
    st.sidebar.success("Prediction generated and added to history.")

filtered_oly = olympic_df[(olympic_df['Year'] >= year_start)].copy()
if event_filter:
    filtered_oly = filtered_oly[filtered_oly['Event'].isin(event_filter)].copy()
if sex_filter != 'All':
    filtered_oly = filtered_oly[filtered_oly['Sex'] == sex_filter].copy()

filtered_rec = records_df[(records_df['Year'] >= year_start)].copy()
if event_filter:
    filtered_rec = filtered_rec[filtered_rec['Event'].isin(event_filter)].copy()
if sex_filter != 'All':
    filtered_rec = filtered_rec[filtered_rec['Sex'] == sex_filter].copy()

tabs = st.tabs(["Introduction & Data Preparation", "Data Preparation Process", "Exploratory Data Analysis", "Statistical Summaries & Insights", "Predictive Modeling", "Documentation & Usage"])

with tabs[0]:
    st.header("Introduction & Data Preparation")
    st.write("This app analyzes historical Olympic and world record swimming data to show trends in performance, equity, and technology. The app shows ~3086 Olympic and ~2327 Records rows for new insights.")
    
    st.subheader("Data Collection & Advanced Preparation")
    st.write("**Data Sources:** Olympic swimming data (1912-2020) and world records data.")
    st.write("**Cleaning:** Outlier removal using IQR, imputation for missing values, time parsing, and country decoding.")
    st.write("**Feature Engineering:** Standardization, label encoding, and polynomial features.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Olympic Rows", filtered_oly.shape[0])
    with col_b:
        st.metric("Records Rows", filtered_rec.shape[0])
    st.metric("Total Rows", len(filtered_oly) + len(filtered_rec))
    

with tabs[1]:
    st.header("Data Preparation Process")
    st.write("Overview of data cleaning and processing steps with visualizations.")
    
    prep_data = load_and_prepare_data()
    (orig_oly, orig_rec, before_out_oly, after_out_oly, before_out_rec, after_out_rec,
     dupe_drop_oly, time_clean_oly, sec_clean_oly, dupe_drop_rec, time_clean_rec, sec_clean_rec,
     miss_oly, miss_rec, figs_h, oly_sum, rec_sum, hist_fig, miss_pct, yrs_miss, yrs_full, orig_sd, mean_sd, med_sd, ffill_sd, bad_times,
     final_oly, final_rec, olympic_df_original, records_df_original) = prep_data
    
    st.subheader("1. Loading and Initial Mappings")
    st.metric("Original Olympic Rows", orig_oly)
    st.metric("Original Records Rows", orig_rec)
    st.write("Mapped genders (M to Men, W to Women), strokes, and added source identifiers.")
    
    st.subheader("2. Duplicates Removal")
    dupe_df = pd.DataFrame({
        "Stage": ["Original", "After Dup Drop", "Null Time Drop", "Parse Fail Drop"],
        "Olympic": [orig_oly, dupe_drop_oly, time_clean_oly, sec_clean_oly],
        "Records": [orig_rec, dupe_drop_rec, time_clean_rec, sec_clean_rec]
    })
    fig_dupe = px.bar(dupe_df, x="Stage", y=["Olympic", "Records"], title="Row Reduction by Cleaning Stage", barmode="group")
    st.plotly_chart(fig_dupe)
    st.write("Tracks data rows removed at each step: duplicates, nulls, and parsing failures.")
    
    st.subheader("3. Missing Values Overview")
    st.write("Heatmaps visualize missing data patterns (yellow is missing).")
    st.pyplot(figs_h[0])
    
    st.subheader("4. Missing Patterns")
    fig_miss = px.bar(x=miss_pct.index, y=miss_pct.values, title="Missing % by Year (Olympic)", labels={"x": "Year", "y": "Missing %"})
    st.plotly_chart(fig_miss)
    st.write(f"Average year with missing vs. full data: {yrs_miss:.1f} vs {yrs_full:.1f}. Significant gap suggests MAR patterns.")
    
    st.subheader("5. Imputation Effectiveness")
    imp_df = pd.DataFrame({
        "Method": ["Original", "Mean Fill", "Median Fill", "Forward Fill"],
        "SD": [orig_sd, mean_sd, med_sd, ffill_sd]
    })
    fig_imp = px.line(imp_df, x="Method", y="SD", title="Std Deviation Changes from Imputation", markers=True)
    st.plotly_chart(fig_imp)
    st.write("Forward fill minimizes bias by preserving temporal patterns.")
    
    st.subheader("6. Time Parsing Failures")
    st.metric("Bad Time Strings Removed", len(bad_times))
    st.write(f"Examples of invalid times: {list(bad_times[:5])}")
    
    st.subheader("7. Outlier Removal (IQR)")
    out_df = pd.DataFrame({
        "Dataset": ["Olympic", "Records"],
        "Before Outliers": [before_out_oly, before_out_rec],
        "After Outliers": [after_out_oly, after_out_rec]
    })
    fig_out = px.bar(out_df, x="Dataset", y=["Before Outliers", "After Outliers"], title="Rows Before/After IQR Outlier Removal", barmode="group")
    st.plotly_chart(fig_out)
    st.metric("Olympic Outliers Removed", before_out_oly - after_out_oly)
    st.metric("Records Outliers Removed", before_out_rec - after_out_rec)
    
    st.write("Time Distribution Changes After Outlier Removal:")
    col_bp1, col_bp2 = st.columns(2)
    with col_bp1:
        st.write("Olympic:")
        fig_bp_oly = px.box(x=olympic_df['Time_sec'], title="Olympic Time_sec (Post-Outlier Removal)")
        st.plotly_chart(fig_bp_oly)
    with col_bp2:
        st.write("Records:")
        fig_bp_rec = px.box(x=records_df['Time_sec'], title="Records Time_sec (Post-Outlier Removal)")
        st.plotly_chart(fig_bp_rec)
    
    st.subheader("8. Final Summaries")
    st.dataframe(oly_sum.round(1))
    st.dataframe(rec_sum.round(1))
    st.pyplot(hist_fig.figure)
    
    st.subheader("Final Rows")
    st.metric("Olympic Final Rows", final_oly)
    st.metric("Records Final Rows", final_rec)

with tabs[2]:
    st.header("Exploratory Data Analysis & Visualizations")
    combined_df = pd.concat([filtered_oly.assign(Source='Olympic'), filtered_rec.assign(Source='Records')])

    if len(combined_df) > 0:
        st.subheader(f"Visualization: {viz_type}")
        if viz_type == 'Histogram':
            fig = px.histogram(combined_df, x='Time_sec', color='Source', nbins=50, title="Time Distribution")
        elif viz_type == 'Box Plot':
            fig = px.box(combined_df, x='Sex', y='Time_sec', color='Source', title="Sex-Time Variance")
        elif viz_type == 'Scatter':
            fig = px.scatter(combined_df, x='Year', y='Time_sec', color='Sex', size='Time_sec', title="Year vs Time")
        elif viz_type == 'Heatmap':
            corr = combined_df[['Year', 'Time_sec', 'Sex_encoded', 'Stroke_encoded']].corr().round(2)
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Viridis')
            fig.update_layout(title_text="Correlation Heatmap")
        elif viz_type == '3D Scatter':
            men_avg = combined_df[combined_df['Sex'] == 'Men'].groupby('Year')['Time_sec'].mean()
            women_avg = combined_df[combined_df['Sex'] == 'Women'].groupby('Year')['Time_sec'].mean()
            gap_series = (men_avg - women_avg).fillna(0)
            combined_df_gap = combined_df.merge(gap_series.to_frame('Gender_Gap'), on='Year').copy()
            fig = px.scatter_3d(combined_df_gap, x='Year', y='Time_sec', z='Gender_Gap', color='Source', hover_data=['Event'], title="Gender Equity")
        elif viz_type == 'Trend Line':
            fig = px.line(combined_df.groupby(['Year', 'Event'], as_index=False)['Time_sec'].mean(), x='Year', y='Time_sec', color='Event', title="Event Trends")

        st.plotly_chart(fig, use_container_width=True)
        st.write("Interactive visualization of six types for data exploration. Histogram and trend line to view progression, box plot and scatter to view gender differences. The 3D scatter shows the difference between the genders in a subtracted gap to show the upward trend for women throughout the years (women getting better faster than men).")
    else:
        st.write("No data matches current filters.")

with tabs[3]:
    st.header("Statistical Summaries & Insights")
    combined_df = pd.concat([filtered_oly.assign(Source='Olympic'), filtered_rec.assign(Source='Records')])

    if len(combined_df) > 0:
        st.subheader("Descriptive Statistics")
        stats_summary = combined_df.groupby('Source')[['Year', 'Time_sec']].describe().T.round(1)
        st.dataframe(stats_summary, use_container_width=True)
        
        global_avg = combined_df['Time_sec'].mean()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Global Average Time (s)", f"{global_avg:.1f}")
        with col2:
            st.metric("Notable Improvement", "100m Freestyle (~10s)")
        
        st.subheader("Statistical Tests")
        corr_yr_time, p_yr_time = pearsonr(combined_df['Year'], combined_df['Time_sec'])
        st.write(f"Year-Time Correlation: r = {corr_yr_time:.3f} (p = {p_yr_time:.3f})")
        
        men_times = combined_df[combined_df['Sex'] == 'Men']['Time_sec']
        women_times = combined_df[combined_df['Sex'] == 'Women']['Time_sec']
        stat, p_gender = ttest_ind(men_times, women_times)
        st.write(f"Gender Gap T-test: t = {stat:.3f} (p = {p_gender:.3f})")
        
        st.subheader("Real-World Insights")
        pre_ban = combined_df[combined_df['Year'] <= 2009]['Time_sec'].mean()
        post_ban = combined_df[combined_df['Year'] > 2009]['Time_sec'].mean()
        percent_change = ((post_ban - pre_ban) / pre_ban) * 100
        st.write(f"Tech Suit Ban Impact: Pre-2010: {pre_ban:.2f}s → Post-2010: {post_ban:.2f}s ({percent_change:.2f}% change). Indicates plateau in performance.")
        st.write("Feature Engineering: Polynomial terms for trends, statistical aggregations.")
    else:
        st.write("Adjust filters for results.")

with tabs[4]:
    st.header("Predictive Modeling")
    st.write("Machine learning models trained on year, sex, and stroke for time predictions.")
    
    st.subheader("Model Performance")
    col6, col7 = st.columns(2)
    with col6:
        st.write("Olympic Models:")
        st.dataframe(pd.DataFrame(results_oly).T.round(3), use_container_width=True)
        st.write("Random Forest performs best.")
    with col7:
        st.write("Records Models:")
        st.dataframe(pd.DataFrame(results_rec).T.round(3), use_container_width=True)
        st.write("Random Forest performs best.")
    
    if st.session_state['latest_prediction']:
        latest = st.session_state['latest_prediction']
        st.subheader("Latest Prediction")
        st.write(f"Predicted for {latest['Year']} {latest['Sex']} {latest['Stroke']} ({latest['Source']} data):")
        st.write(f"Linear Regression: {latest['LinReg']:.2f}s")
        st.write(f"Random Forest: {latest['RF']:.2f}s")
        st.write(f"Gradient Boosting: {latest['GB']:.2f}s")
        pred_df = pd.DataFrame({'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'], 'Predicted Time (s)': [latest['LinReg'], latest['RF'], latest['GB']]})
        fig_pred_bar = px.bar(pred_df, x='Model', y='Predicted Time (s)', title="Prediction Comparison", color='Model', text_auto='.2f')
        st.plotly_chart(fig_pred_bar, use_container_width=True)
    
    st.subheader("Prediction History")
    if st.session_state['prediction_history']:
        history_df = pd.DataFrame(st.session_state['prediction_history'][-5:])
        st.dataframe(history_df, use_container_width=True)
        fig_pred = px.line(history_df, x='Year', y='RF', color='Sex', markers=True, title="Random Forest Trend")
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.write("No predictions yet.")

    st.write("Ensemble methods enhance prediction accuracy.")

with tabs[5]:
    st.header("Documentation & Usage")
    st.write("Explore tabs with sidebar filters and prediction tools for dynamic insights.")
    st.write("**Repository:** https://github.com/BrooklynHall/midterm-cmse-hall")
    st.write("**Data Dictionary:** Year (int), Time_sec (float), Sex (str: Men/Women), Stroke (str), Event (str), Nationality (str).")


st.sidebar.text("Blake Hall - CMSE 830 Final Project")
