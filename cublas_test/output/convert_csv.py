import pandas as pd
import re

# Read the CSV file
with open('result.csv', 'r') as f:
    lines = f.readlines()

# Parse the data
data = []
current_config = None

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    if line.startswith('Config Name,'):
        # Extract N and k from configuration header
        config_part = line.split(',')[1]  # Get 'N=xxxx_k=yyyy'
        match = re.match(r'N=(\d+)_k=(\d+)', config_part)
        if match:
            current_config = {'N': int(match.group(1)), 'k': int(match.group(2))}
    else:
        # Parse data line
        parts = line.split(',')
        if len(parts) == 3 and current_config:
            gemm_type, metric_name, value = parts
            data.append({
                'N': current_config['N'],
                'k': current_config['k'],
                'GEMM_Type': gemm_type,
                'metric_name': metric_name,
                'value': float(value)
            })

# Create DataFrame
df = pd.DataFrame(data)

# Pivot to wide format
df_wide = df.pivot_table(
    index=['N', 'k', 'GEMM_Type'], 
    columns='metric_name', 
    values='value', 
    aggfunc='first'
).reset_index()

# Flatten column names
df_wide.columns.name = None

# Write to a new file first
df_wide.to_csv('result_wide.csv', index=False)

print('Data conversion completed successfully!')
print(f'Shape: {df_wide.shape}')
print(f'Total configurations: {len(df_wide)}')
print(f'Unique N values: {sorted(df_wide["N"].unique())}')
print(f'Unique k values: {sorted(df_wide["k"].unique())}')
print(f'GEMM types: {list(df_wide["GEMM_Type"].unique())}')
print(f'Number of metrics: {len(df_wide.columns) - 3}')
print('\nFirst few columns:', list(df_wide.columns)[:10])