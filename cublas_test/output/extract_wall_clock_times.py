#!/usr/bin/env python3
import re
import pandas as pd

def extract_timing_data(txt_file):
    """Extract timing data from comparison.txt file"""
    timing_data = []
    
    with open(txt_file, 'r') as f:
        content = f.read()
    
    # Split content by test runs (commands)
    test_runs = re.split(r'\[.*?\] Executing:', content)
    
    for test_run in test_runs[1:]:  # Skip first empty split
        # Extract N and k parameters from command
        cmd_match = re.search(r'./cublas_test\s+(\d+)\s+(\d+)', test_run)
        if not cmd_match:
            continue
            
        N, k = int(cmd_match.group(1)), int(cmd_match.group(2))
        
        # Find all timing results in this test run
        # Pattern: Test Type followed by time value
        timing_patterns = [
            (r'Big GEMM\s+([0-9.]+)', 'BigGEMM'),
            (r'Naive GEMM\s+([0-9.]+)', 'NaiveGEMM'),
            (r'Batched GEMM\s+([0-9.]+)', 'BatchedGEMM'),
            (r'Strided Batch GEMM\s+([0-9.]+)', 'StridedBatchedGEMM')
        ]
        
        for pattern, gemm_type in timing_patterns:
            matches = re.findall(pattern, test_run)
            if matches:
                time_ms = float(matches[0])  # Take first match
                timing_data.append({
                    'N': N,
                    'k': k,
                    'GEMM_Type': gemm_type,
                    'wall_clock_time': time_ms
                })
    
    return timing_data

def main():
    # Extract timing data from comparison.txt
    txt_file = '/home/ubuntu/dbs/llm/llm.c/cublas_test/comparison.txt'
    timing_data = extract_timing_data(txt_file)
    
    # Load existing CSV
    csv_file = '/home/ubuntu/dbs/llm/llm.c/cublas_test/output/result_wide.csv'
    df = pd.read_csv(csv_file)
    
    # Create a DataFrame from timing data for easier merging
    timing_df = pd.DataFrame(timing_data)
    
    # Merge timing data with existing CSV based on N, k, and GEMM_Type
    df_merged = df.merge(timing_df, on=['N', 'k', 'GEMM_Type'], how='left')
    
    # Save the updated CSV
    output_file = '/home/ubuntu/dbs/llm/llm.c/cublas_test/output/result_wide_with_timing.csv'
    df_merged.to_csv(output_file, index=False)
    
    print(f"Updated CSV saved to: {output_file}")
    print(f"Added wall_clock_time column with {len(timing_data)} timing entries")
    
    # Show some sample data to verify
    print("\nSample of merged data:")
    print(df_merged[['N', 'k', 'GEMM_Type', 'wall_clock_time']].head(10))

if __name__ == "__main__":
    main()