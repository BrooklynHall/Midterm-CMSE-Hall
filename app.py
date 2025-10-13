import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# load cleaned data
olympic_df = pd.read_csv('data/cleaned_olympic_data.csv')
records_df = pd.read_csv('data/cleaned_records_data.csv')

# advanced cleaning using outlier removal by IQR 
def remove_outliers(df, col):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
olympic_df = remove_outliers(olympic_df, 'Time_sec')
records_df = remove_outliers(records_df, 'Time_sec')

st.title("Swimming Performance Explorer")
st.sidebar.header("Filters")
year_start = st.sidebar.slider("Start Year", 1912, 2020, 1912)
event_filter = st.sidebar.multiselect("Events", sorted(olympic_df['Event'].unique()), [])
sex_filter = st.sidebar.selectbox("Sex", ['All', 'Men', 'Women'])
source_filter = st.sidebar.selectbox("Source", ['All', 'Olympic', 'Records'])

# filter
filtered_olympic = olympic_df[(olympic_df['Year'] >= year_start)].copy()
if event_filter:
    filtered_olympic = filtered_olympic[filtered_olympic['Event'].isin(event_filter)].copy()
if sex_filter != 'All':
    filtered_olympic = filtered_olympic[filtered_olympic['Sex'] == sex_filter].copy()
filtered_records = records_df[(records_df['Year'] >= year_start)].copy()
if event_filter:
    filtered_records = filtered_records[filtered_records['Event'].isin(event_filter)].copy()
if sex_filter != 'All':
    filtered_records = filtered_records[filtered_records['Sex'] == sex_filter].copy()

# tabs
tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Exploration", "Insights & Analysis", "Documentation"])

with tab1:
    st.header("Swimming Performance: Olympic vs World Records Over Time")
    st.write("This app goes through swimming performance trends using Olympic and world records data, including looks into gender differences, country performances, event-specific improvements, and factors like the 2009 techsuit ban.")
    avg_olympic = filtered_olympic['Time_sec'].mean() if len(filtered_olympic) > 0 else np.nan
    avg_records = filtered_records['Time_sec'].mean() if len(filtered_records) > 0 else np.nan
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Olympic Time (seconds)", f"{avg_olympic:.1f}" if not np.isnan(avg_olympic) else "No data")
    col2.metric("Average Records Time (seconds)", f"{avg_records:.1f}" if not np.isnan(avg_records) else "No data")
    quick_combined = pd.concat([filtered_olympic.assign(Source='Olympic'), filtered_records.assign(Source='Records')])
    if len(quick_combined) > 0:
        fastest_idx = quick_combined['Time_sec'].idxmin()
        fastest_event = str(quick_combined.loc[fastest_idx, 'Event']) if pd.notna(fastest_idx) else "No data"
        if len(fastest_event) > 20:
            fastest_event = fastest_event[:17] + "..."
        col3.metric("Fastest Event Overall", fastest_event)
    else:
        col3.metric("Fastest Event Overall", "No data")
    st.subheader("Data Sources & Cleaning Overview")
    st.write("Data from two sources: Olympic results and world records. Cleaned with imputation, forward-fill after comparison, outlier removal, and standardization. MAR missing patterns detected and handled.")

with tab2:
    st.header("Trends Exploration")
    # build combined_df here for tab2-specific use
    if source_filter == 'Olympic':
        combined_df = filtered_olympic.assign(Source='Olympic').copy()
    elif source_filter == 'Records':
        combined_df = filtered_records.assign(Source='Records').copy()
    else:
        combined_df = pd.concat([filtered_olympic.assign(Source='Olympic'), filtered_records.assign(Source='Records')]).copy()
    
    # per Year aggregated gender gap men - women, using mean groupby
    men_per_year = combined_df[combined_df['Sex'] == 'Men'].groupby('Year')['Time_sec'].mean()
    women_per_year = combined_df[combined_df['Sex'] == 'Women'].groupby('Year')['Time_sec'].mean()
    gap_series = (men_per_year - women_per_year).fillna(0)
    gap_df = gap_series.to_frame('Gender_Gap').reset_index()
    combined_df = combined_df.merge(gap_df, on='Year', how='left')
    
    if len(combined_df) > 0 and not combined_df['Time_sec'].isna().all():
        st.subheader("Time Trends by Year")
        st.write("This line chart displays average swim times across years, separated by data source. Olympic trends show overall progress, while records can jump due to event variety. Use the sidebar filters to see more on specific events!")
        summary = combined_df.groupby(['Year', 'Source'])['Time_sec'].mean().reset_index()
        fig = px.line(summary, x='Year', y='Time_sec', color='Source')
        st.plotly_chart(fig, key="time_trends_line")
        st.subheader("Bar Chart: Top Countries")
        st.write("This bar chart ranks the top 10 countries by Olympic performance count. Percentages indicate their proportion of total results.")
        top_countries = filtered_olympic['Nationality_full'].value_counts().head(10).reset_index()
        top_countries.columns = ['Nationality', 'Count']
        total = top_countries['Count'].sum()
        top_countries['Percentage'] = (top_countries['Count'] / total * 100).round(1).astype(str) + '%'
        fig_bar = px.bar(top_countries, x='Count', y='Nationality', orientation='h', color='Count')
        st.plotly_chart(fig_bar, key="top_countries_bar")
        st.subheader("Interactive 3D Scatter Plot: Year vs Time vs Gender Gap")
        st.write("This 3D scatter plot examines gender equity in swimming performance: year on the x-axis, time on the y-axis, and the z-axis shows the per-year gender gap (average men's times minus women's times). As the years progress, the gap often narrows showing that women's swim is catching up!")
        fig_3d = px.scatter_3d(combined_df, x='Year', y='Time_sec', z='Gender_Gap', color='Source', hover_data=['Sex', 'Event'])
        st.plotly_chart(fig_3d, key="gender_gap_3d")
        st.subheader("Gender Performance Differences")
        st.write("This bar chart compares average times by gender and data source, showing historical gaps that are narrowing over time.")
        gender_stats = combined_df.groupby(['Source', 'Sex'])['Time_sec'].mean().reset_index()
        fig_gender = px.bar(gender_stats, x='Sex', y='Time_sec', color='Source')
        st.plotly_chart(fig_gender, key="gender_diff_bar")
    else:
        st.write("No data available—adjust filters.")

with tab3:
    st.header("Deeper Insights")
    if source_filter == 'Olympic':
        combined_df_insights = filtered_olympic.assign(Source='Olympic').copy()
    elif source_filter == 'Records':
        combined_df_insights = filtered_records.assign(Source='Records').copy()
    else:
        combined_df_insights = pd.concat([filtered_olympic.assign(Source='Olympic'), filtered_records.assign(Source='Records')]).copy()
    
    if len(combined_df_insights) > 0 and not combined_df_insights['Time_sec'].isna().all():
        st.subheader("Event-Specific Trends")
        st.write("This standout line chart shows median swim times over years for each event, using sidebar selections or all data if none. It shows progress in specific strokes and distances.")
        if len(combined_df_insights) > 0:
            fig_event = px.line(combined_df_insights.groupby(['Year', 'Event'])['Time_sec'].median().reset_index(), x='Year', y='Time_sec', color='Event')
            st.plotly_chart(fig_event, key="event_specific_line")
        st.subheader("2009 Swimsuit Ban Impact")
        st.write("This line chart analyzes post-2009 data to show the techsuit ban's effect, where Olympic times slowed temporarily compared to continuing records improvements.")
        post_ban = combined_df_insights[combined_df_insights['Year'] > 2009].copy()
        if len(post_ban) > 0:
            fig_ban = px.line(post_ban.groupby(['Year', 'Source'])['Time_sec'].mean().reset_index(), x='Year', y='Time_sec', color='Source')
            st.plotly_chart(fig_ban, key="ban_impact_line")
    else:
        st.write("No data available—adjust filters.")


with tab4:
    st.header("Documentation:")
    st.write("**How to Use:** Use sidebar filters for custom views tabs organize content.")
    st.write("**Data Sources:** Kaggle Olympic data, GitHub Records.")
    st.write("https://www.kaggle.com/datasets/datasciencedonut/olympic-swimming-1912-to-2020")
    st.write("https://github.com/n-reeves/Swim-Record-Analysis")
    st.write("**Github Repo:** https://github.com/BrooklynHall/Midterm-CMSE-Hall")

st.sidebar.text("Blake Hall - CMSE 830 Midterm")