import pandas as pd
import numpy as np
import pycountry
import matplotlib.pyplot as plt
import seaborn as sns

# load olympic data
olympic_df = pd.read_csv('data/olympic_swimming_1912_2020.csv')
olympic_df.rename(columns={'Team': 'Nationality', 'Results': 'Time', 'Gender': 'Sex', 'Distance (in meters)': 'Distance'}, inplace=True)

sex_map = {'M': 'Men', 'W': 'Women'}
olympic_df['Sex'] = olympic_df['Sex'].map(sex_map).fillna(olympic_df['Sex'])
olympic_df['Distance'] = olympic_df['Distance'].astype(str).str.replace('m', '') + 'm'
olympic_df['Event'] = olympic_df['Distance'] + ' ' + olympic_df['Stroke']
olympic_df['Source'] = 'Olympic'

# load records data
records_df = pd.read_csv('data/swim_records.csv', encoding='cp1252')
# MAP strokes to match olympic full names
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

print("Olympic loaded:", olympic_df.shape)
print("Records loaded:", records_df.shape)

# missing totals
print("Missing in Olympic total:", olympic_df.isnull().sum().sum())
print("Missing in Records total:", records_df.isnull().sum().sum())

# missing heatmap olympic
plt.figure(figsize=(12,6))
df_sorted = olympic_df.sort_values('Year', ascending=True)
sns.heatmap(df_sorted.isna(), cmap="magma")
plt.title("Missing - Olympic")
plt.show()

# missing heatmap records
plt.figure(figsize=(12,6))
df_sorted_rec = records_df.sort_values('Date', ascending=True)
sns.heatmap(df_sorted_rec.isna(), cmap="magma")
plt.title("Missing - Records")
plt.show()

# basic summaries
print("Olympic summary:")
print(olympic_df.describe())
print("Records summary:")
print(records_df.describe())

# histogram years
olympic_df['Year'].hist(bins=20)
plt.title("Year Distribution - Olympic")
plt.show()

# advanced data processing complex missing patterns first
print("Complex patterns check before impute")
olympic_df['missing_raw_time'] = olympic_df['Time'].isnull()
print("Missing by year %:", olympic_df.groupby('Year')['missing_raw_time'].mean())
years_missing = olympic_df[olympic_df['missing_raw_time']]['Year'].mean()
years_full = olympic_df[~olympic_df['missing_raw_time']]['Year'].mean()
print(f"Avg year missing vs full: {years_missing} vs {years_full}")
if pd.isna(years_missing):
    print("No missing raw Time—MCAR random, no patterns")
elif abs(years_missing - years_full) < 5:
    print("Looks MCAR random")
else:
    print("Looks MAR missing related to year")

# clean olympic
cleaned_oly = olympic_df.copy()
cleaned_oly = cleaned_oly[['Year', 'Distance', 'Stroke', 'Sex', 'Nationality', 'Athlete', 'Time', 'Rank', 'Event', 'Source']]
cleaned_oly.drop_duplicates(inplace=True)
cleaned_oly = cleaned_oly[cleaned_oly['Time'].notnull()]
cleaned_oly['Time'] = cleaned_oly['Time'].astype(str)

def time_seconds(s):
    try:
        s_str = str(s).strip()
        # If no colon, parse as decimal seconds directly 
        if ':' not in s_str:
            return float(s_str)
        # If colon, parse mm:ss.decimal or hh:mm:ss.decimal
        parts = s_str.split(':')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].replace('.', '').isdigit():
            # mm:ss.decimal
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3 and all(p.replace('.', '').isdigit() for p in parts):
            # hh:mm:ss.decimal
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return (hours * 3600) + (minutes * 60) + seconds
        else:
            return np.nan  # invalid format
    except (ValueError, IndexError):
        return np.nan
    return np.nan

cleaned_oly['Time_sec'] = cleaned_oly['Time'].apply(time_seconds)

bad_times = cleaned_oly[cleaned_oly['Time_sec'].isna()]['Time'].unique()
print("Bad Time strings", bad_times[:10])
print("NaN in Time_sec before impute", cleaned_oly['Time_sec'].isna().sum())

# multiple imputations and compare effectiveness 
oly_test = cleaned_oly['Time_sec'].copy()
oly_test.iloc[::10] = np.nan  # 10% missing

def impute_mean(col):
    return col.fillna(col.mean())

def impute_median(col):
    return col.fillna(col.median())

def impute_ffill(col):
    return col.fillna(method='ffill')

time_mean = impute_mean(oly_test.copy())
time_median = impute_median(oly_test.copy())
time_ffill = impute_ffill(oly_test.copy())

original_sd = oly_test.std()
mean_sd = time_mean.std()
median_sd = time_median.std()
ffill_sd = time_ffill.std()
print("Imputation effectiveness:")
print(f"Original SD: {original_sd}")
print(f"Mean SD: {mean_sd}, Change: {abs(mean_sd - original_sd)}")
print(f"Median SD: {median_sd}, Change: {abs(median_sd - original_sd)}")
print(f"FFill SD: {ffill_sd}, Change: {abs(ffill_sd - original_sd)}")

# actual impute use ffill
cleaned_oly['Time_sec'] = impute_ffill(cleaned_oly['Time_sec'])
cleaned_oly['Time_sec'] = cleaned_oly['Time_sec'].fillna(cleaned_oly['Time_sec'].mean())

cleaned_oly = cleaned_oly[cleaned_oly['Time_sec'].notnull()]

# decode country
def decode(c):
    try:
        return pycountry.countries.get(alpha_3=c).name
    except:
        return c
cleaned_oly['Nationality_full'] = cleaned_oly['Nationality'].apply(decode)

# clean records
cleaned_rec = records_df.copy()
cleaned_rec['Event'] = cleaned_rec.apply(lambda row: f"{row['Distance']}m {row['Stroke']}", axis=1)
if 'Year' not in cleaned_rec.columns:
    cleaned_rec['Year'] = cleaned_rec['Date'].str.extract(r'(\d{4})', expand=False).astype(float)
if 'Athlete' not in cleaned_rec.columns:
    cleaned_rec['Athlete'] = cleaned_rec['Swimmer']
cleaned_rec = cleaned_rec[['Year', 'Distance', 'Stroke', 'Sex', 'Nationality', 'Athlete', 'Time', 'Event', 'Source']]
cleaned_rec.drop_duplicates(inplace=True)
cleaned_rec = cleaned_rec[cleaned_rec['Time'].notnull()]
cleaned_rec['Time'] = cleaned_rec['Time'].astype(str)
cleaned_rec['Time_sec'] = cleaned_rec['Time'].apply(time_seconds)
cleaned_rec = cleaned_rec[cleaned_rec['Time_sec'].notnull()]
cleaned_rec['Time_sec'] = impute_ffill(cleaned_rec['Time_sec'])
cleaned_rec['Time_sec'] = cleaned_rec['Time_sec'].fillna(cleaned_rec['Time_sec'].mean())
cleaned_rec['Nationality_full'] = cleaned_rec['Nationality'].apply(decode)

# save cleaned
cleaned_oly.to_csv('data/cleaned_olympic_data.csv', index=False)
cleaned_rec.to_csv('data/cleaned_records_data.csv', index=False)