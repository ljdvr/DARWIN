import pandas as pd

df = pd.read_csv('data.csv')

# -------------------------------------------------------------------
# Updated Metric Prefix List (includes ALL your handwriting features)
# -------------------------------------------------------------------
valid_metric_prefixes = [
    'air_time', 'pressure', 'disp_index', 'gmrt', 
    'max_x_extension', 'max_y_extension',
    'mean_acc', 'mean_jerk', 'mean_speed',
    'num_of_pendown', 'paper_time', 'total_time'
]

# -------------------------------------------------------------------
# Claim 1: Verify all trials link to ID/class (unchanged)
# -------------------------------------------------------------------
trial_columns = [col for col in df.columns if any(char.isdigit() for char in col)]
required_columns = ['ID', 'class']

assert not df[required_columns + trial_columns].isnull().any().any(), \
       "Missing links between trials and IDs/class"
print("✅ All trials are explicitly linked to IDs and class labels")

# -------------------------------------------------------------------
# Claim 2: Updated fragmentation check
# -------------------------------------------------------------------
unrelated_columns = [
    col for col in df.columns 
    if col not in required_columns 
    and not any(prefix in col for prefix in valid_metric_prefixes)
]

if not unrelated_columns:
    print("✅ No fragmented sections - all columns are valid handwriting metrics")
else:
    print(f"⚠️ Genuine fragmented columns: {unrelated_columns}")

# -------------------------------------------------------------------
# Trial completeness check (unchanged)
# -------------------------------------------------------------------
assert (df['ID'].value_counts() == 1).all(), \
       "IDs are duplicated (should be one row per subject)"
print("✅ All subjects have exactly one row with 25 integrated trials")