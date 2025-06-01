import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load cleaned wide-format data
df = pd.read_csv('DARWIN_cleaned.csv')

# ----- 1. Dimensionality Reduction -----
# Drop near-zero variance columns
variances = df.var(numeric_only=True)
low_variance_cols = variances[variances < 1e-3].index.tolist()
df.drop(columns=low_variance_cols, inplace=True)

# Drop highly correlated features (correlation > 0.95)
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
df.drop(columns=high_corr_cols, inplace=True)

# ----- 2. Numerosity Reduction -----
# Round numeric columns to 1 decimal place
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].round(1)

# ----- 3. Data Transformation: Normalization -----
scaler = StandardScaler()
scaled_cols = df.select_dtypes(include=[np.number]).columns
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# ----- 4. Concept Hierarchy Generation -----
# Aggregate all trial features per subject (ID)
if 'ID' in df.columns:
    df_grouped = df.groupby('ID').mean(numeric_only=True).reset_index()
else:
    df_grouped = df.copy()  # fallback if ID column is missing

# Save to a single standardized file
df_grouped.to_csv('DARWIN_standardized.csv', index=False)
print("✅ Preprocessing complete. File saved as 'DARWIN_standardized.csv'")
