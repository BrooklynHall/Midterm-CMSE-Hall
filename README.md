**Blake Hall**

https://midterm-cmse-hall.streamlit.app

**Project Overview**

This project explores swimming performance trends using historical Olympic results (1912-2020) and world records data. It provides plots to analyze time improvements, gender differences, top performing countries, event specific trends, and the impact of the 2009 techsuit ban.



I chose this dataset due to my 13 long years of highly competitive swimming and knowledge of the swim competitive scene. Swimming was a big portion of my life, so I wanted to see some data on it and explore what I used to spend all my time at. Both the datasets were good at showing this relationship, particularly the Olympic data from Kaggle was easy to work with, but both of them allow good looks at the progression of the sport.



From the ida/eda we see the trend of year and time, where time goes down and improvements are made gradually over time, dipping a little in 2009. We see that there is a gender aspect to the sport that is decreasing over time, women are slowly catching up to men (particularly Katie Ledecky). There were some outlier times that were from the old Olympic data (1920's or so) that were removed. We ended up removing some of the outliers via IQR (although most wouldn't matter much anyways). We see that faster events have actually tended to improve faster than longer distance races, mostly due to technique changes.



For preprocessing we imported a ton of data in through csv, then handled some missing data by testing through 10% missing simulations and ended up using forward fill imputation (which seemed to work the best). We removed outliers that had over a 1.5X range in time. We also standardized the time to seconds so we could see all the data together. The data merge was a big part of this, as both our datasets were labeled differently and handled times/names/events differently.



With streamlit, i tried making a tabbed interface that has sidebar filters that effect every chart. I have made line, bar, 3d scatter, and a few more charts to boot. All of the charts are interactive using plotly.



**Features include:**



**Data Cleaning \& Processing:** Standardized formats (Sex to Men/Women, Event names), time parsing (mm:ss to seconds), imputation for missing values (forward-fill preferred after testing mean and median), outlier removal via IQR.

**Exploratory Data Analysis (EDA):** Heatmaps for missing patterns, histograms, summaries, Missing At Random related to year.

**Visualizations**: Line charts for time trends, bar charts for countries/gender, event-specific medians.

**Streamlit App:** Interactive filters (year, events, sex, source) across tabs for introduction, trends, insights, documentation.

**Insights:** Shows tech ban slowing Olympic improvements post-2009 with steady records progress.

The code uses plotly for plotting, handling 5,000 rows of data from two sources (Olympic 4311 rows, Records 1683 rows).



**Setup**:

Needs the following libraries:

pandas numpy pycountry matplotlib seaborn streamlit plotly



**File Setup:** 

data/olympic\_swimming\_1912\_2020.csv

data/swim\_records.csv

data\_prep.py

app.py



**To clean run data\_prep.py**

**To run app: streamlit run app.py**

