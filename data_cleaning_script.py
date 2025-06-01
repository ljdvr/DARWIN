import pandas as pd
import numpy as np
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# Check for extreme values in time variables
time_cols = [col for col in df.columns if 'time' in col.lower()]
print("\nTime variable statistics:")
print(df[time_cols].describe())

# Check for negative values in measurements (shouldn't exist)
numeric_cols = df.select_dtypes(include=[np.number]).columns
negative_check = (df[numeric_cols] < 0).any()
print("\nColumns with negative values:")
print(negative_check[negative_check])

# Drop rows with missing values
df = df.dropna()

# ----------------------------------------
# Data Transformation: Normalization
# ----------------------------------------
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['class'], errors='ignore')
scaler = StandardScaler()
normalized_data = scaler.fit_transform(numeric_df)
normalized_df = pd.DataFrame(normalized_data, columns=numeric_df.columns)

# ----------------------------------------
# Dimensionality Reduction + Compression: PCA
# ----------------------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]
print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)

# ----------------------------------------
# Numerosity Reduction: KMeans Clustering
# ----------------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(normalized_data)
print("\nCluster value counts:\n", df['cluster'].value_counts())

# ----------------------------------------
# Data Discretization: Binning
# ----------------------------------------
discretized_df = normalized_df.copy()
for col in discretized_df.columns:
    discretized_df[col + '_bin'] = pd.cut(discretized_df[col], bins=3, labels=['Low', 'Medium', 'High'])

# ----------------------------------------
# Concept Hierarchy Generation
# ----------------------------------------
concept_hierarchy = {
    0: 'Not Naturalized',
    1: 'Naturalized'
}
if 'class' in df.columns:
    df['class_hierarchy'] = df['class'].map(concept_hierarchy)

# ----------------------------------------
# Optional: Reshape from wide to long format
# ----------------------------------------
metric_prefixes = set()
for col in df.columns:
    if col not in ['ID', 'class', 'cluster', 'PCA1', 'PCA2', 'class_hierarchy']:
        base_name = ''.join([c for c in col if not c.isdigit()])
        metric_prefixes.add(base_name)

trial_numbers = set()
for col in df.columns:
    if any(char.isdigit() for char in col):
        trial_num = ''.join([c for c in col if c.isdigit()])
        trial_numbers.add(trial_num)
trial_numbers = sorted(trial_numbers)

long_dfs = []
for metric in metric_prefixes:
    cols = [col for col in df.columns if col.startswith(metric)]
    if not cols:
        continue
    temp_df = df.melt(id_vars=['ID', 'class'], 
                      value_vars=cols,
                      var_name='trial_metric',
                      value_name=metric)
    temp_df['trial'] = temp_df['trial_metric'].str.extract('(\d+)')
    temp_df.drop('trial_metric', axis=1, inplace=True)
    long_dfs.append(temp_df)

if long_dfs:
    final_long_df = reduce(lambda left, right: pd.merge(left, right, on=['ID', 'class', 'trial']), long_dfs)
    print("\nLong format data shape:", final_long_df.shape)
    final_long_df.to_csv('DARWIN_long_format.csv', index=False)

# ----------------------------------------
# Save Preprocessed Outputs
# ----------------------------------------
df.to_csv('DARWIN_cleaned.csv', index=False)
discretized_df.to_csv('DARWIN_discretized.csv', index=False)
