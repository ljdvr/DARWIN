import pandas as pd
import numpy as np
from functools import reduce

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

# Remove outliers using IQR method for numeric columns
def remove_outliers(df, columns):
    clean_df = df.copy()
    for col in columns:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    return clean_df

numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = numeric_cols.drop('class') if 'class' in numeric_cols else numeric_cols
df_clean = remove_outliers(df, numeric_cols)
print(f"\nOutliers removed. Rows before: {df.shape[0]}, after: {df_clean.shape[0]}")

# Check for extreme values in time variables
time_cols = [col for col in df_clean.columns if 'time' in col.lower()]
print("\nTime variable statistics:")
print(df_clean[time_cols].describe())

# Check for negative values in measurements (shouldn't exist)
negative_check = (df_clean[numeric_cols] < 0).any()
print("\nColumns with negative values:")
print(negative_check[negative_check])

# Save cleaned data
df_clean.to_csv('DARWIN_cleaned.csv', index=False)