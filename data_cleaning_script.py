import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data.csv')

# Basic data inspection
print("Initial data shape:", df.shape)
print("Missing values per column:\n", df.isnull().sum())

# Check for duplicate IDs
if df['ID'].duplicated().any():
    print("Duplicate IDs found")
else:
    print("No duplicate IDs")

# Convert ID to string if not already
df['ID'] = df['ID'].astype(str)

# Recode class variable: Patient=0, Healthy=1
if 'class' in df.columns:
    df['class'] = df['class'].map({'Patient': 0, 'Healthy': 1})
    print("\nClass variable recoded - Patient:0, Healthy:1")
    print(df['class'].value_counts())

# Check for extreme values in time variables
time_cols = [col for col in df.columns if 'time' in col.lower()]
print("\nTime variable statistics:")
print(df[time_cols].describe())

# Check for negative values in measurements (shouldn't exist)
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = numeric_cols.drop('class') if 'class' in numeric_cols else numeric_cols
negative_check = (df[numeric_cols] < 0).any()
print("\nColumns with negative values:")
print(negative_check[negative_check])

# If we want to reshape the data from wide to long format for analysis:
# This would integrate all the trial data into a more analysis-friendly format

# First, identify the metric prefixes
metric_prefixes = set()
for col in df.columns:
    if col not in ['ID', 'class']:
        # Split by numbers to get the base metric name
        base_name = ''.join([c for c in col if not c.isdigit()])
        metric_prefixes.add(base_name)

# Create a list of all trial numbers present in the data
trial_numbers = set()
for col in df.columns:
    if any(char.isdigit() for char in col):
        trial_num = ''.join([c for c in col if c.isdigit()])
        trial_numbers.add(trial_num)
trial_numbers = sorted(trial_numbers)

# Melt the dataframe into long format
long_dfs = []
for metric in metric_prefixes:
    # Get all columns for this metric across all trials
    cols = [col for col in df.columns if col.startswith(metric)]
    temp_df = df.melt(id_vars=['ID', 'class'], 
                     value_vars=cols,
                     var_name='trial_metric',
                     value_name=metric)
    
    # Extract trial number
    temp_df['trial'] = temp_df['trial_metric'].str.extract('(\d+)')
    temp_df.drop('trial_metric', axis=1, inplace=True)
    
    long_dfs.append(temp_df)

# Merge all long format dataframes
from functools import reduce
final_long_df = reduce(lambda left, right: pd.merge(left, right, on=['ID', 'class', 'trial']), long_dfs)

print("\nLong format data shape:", final_long_df.shape)

# Save cleaned data
df.to_csv('DARWIN_cleaned.csv', index=False)
