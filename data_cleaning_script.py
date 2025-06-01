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
df['class'] = df['class'].map({'P': 0, 'H': 1})

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

# Save cleaned data
df.to_csv('DARWIN_cleaned.csv', index=False)