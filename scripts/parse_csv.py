#!/usr/bin/env python3
"""
Parse GPU metrics CSV file and reorganize into a single table.

Input format:
- Config Name,B=<value>T=<value>
- <range_name>,<metric_name>,<value>
- ...

Output format:
- Single table with columns: Range, B, T, metric1, metric2, ...
"""

import csv
import re
from collections import defaultdict, OrderedDict

def parse_config_name(config_str):
    """Extract B, T, and NH values from config name like 'B=4T=64NH=12'. Falls back to NH=12 if not specified."""
    match = re.match(r'B=(\d+)T=(\d+)(?:NH=(\d+))?', config_str)
    if match:
        b = int(match.group(1))
        t = int(match.group(2))
        nh = int(match.group(3)) if match.group(3) else 12  # Default to 12 if not found
        return b, t, nh
    return None, None, None

def parse_log_file(log_file):
    """
    Parse the log file to extract wall clock time, kernel count, and grid/block sizes.
    Returns:
    - timing_data: dict {(range_name, B, T, NH): {'wall_clock_time': value, 'kernel_count': value}}
    - kernel_configs: dict {(range_name, B, T, NH): [list of (grid, block) tuples]}
    """
    timing_data = {}
    kernel_configs = defaultdict(list)
    
    current_b = None
    current_t = None
    current_nh = None
    current_range = None
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Extract B and T from execution command
            exec_match = re.search(r'Executing:.*-b\s+(\d+)\s+-t\s+(\d+)', line)
            if exec_match:
                current_b = int(exec_match.group(1))
                current_t = int(exec_match.group(2))
                continue
            
            # Extract NH (num_heads) from log output
            nh_match = re.match(r'num_heads:\s+(\d+)', line)
            if nh_match:
                current_nh = int(nh_match.group(1))
                continue
            
            # Extract wall clock time for ranges
            timed_match = re.match(r'\[TIMED\]\s+(\S+)\s+took\s+(\d+)\s+µs', line)
            if timed_match and current_b is not None and current_t is not None:
                range_name = timed_match.group(1)
                wall_clock_time = int(timed_match.group(2))
                # Default to NH=12 if not found yet
                nh = current_nh if current_nh is not None else 12
                if (range_name, current_b, current_t, nh) not in timing_data:
                    timing_data[(range_name, current_b, current_t, nh)] = {}
                timing_data[(range_name, current_b, current_t, nh)]['wall_clock_time'] = wall_clock_time
                continue
            
            # Extract kernel count
            kernel_count_match = re.match(r'Range Name:\s+(\S+),\s+Kernel Count:\s+(\d+)', line)
            if kernel_count_match and current_b is not None and current_t is not None:
                range_name = kernel_count_match.group(1)
                kernel_count = int(kernel_count_match.group(2))
                # Default to NH=12 if not found yet
                nh = current_nh if current_nh is not None else 12
                if (range_name, current_b, current_t, nh) not in timing_data:
                    timing_data[(range_name, current_b, current_t, nh)] = {}
                timing_data[(range_name, current_b, current_t, nh)]['kernel_count'] = kernel_count
                continue
            
            # Extract range name from kernel section
            range_name_match = re.match(r'Range Name:\s+(\S+)', line)
            if range_name_match:
                current_range = range_name_match.group(1)
                continue
            
            # Extract kernel grid and block sizes
            kernel_match = re.search(r'<<<\{(\d+),\s*(\d+),\s*(\d+)\},\s*\{(\d+),\s*(\d+),\s*(\d+)\}\s*>>>', line)
            if kernel_match and current_range and current_b is not None and current_t is not None:
                grid = (int(kernel_match.group(1)), int(kernel_match.group(2)), int(kernel_match.group(3)))
                block = (int(kernel_match.group(4)), int(kernel_match.group(5)), int(kernel_match.group(6)))
                # Default to NH=12 if not found yet
                nh = current_nh if current_nh is not None else 12
                kernel_configs[(current_range, current_b, current_t, nh)].append((grid, block))
                continue
    
    return timing_data, kernel_configs

def parse_csv(input_file):
    """
    Parse the input CSV file.
    Returns a dict: {(range_name, B, T, NH): {metric_name: value}}
    Also returns the ordered list of all metrics.
    """
    data = defaultdict(dict)
    all_metrics = OrderedDict()
    range_order = OrderedDict()
    
    current_config = None
    current_b = None
    current_t = None
    current_nh = None
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            
            if row[0] == 'Config Name':
                # New configuration
                current_config = row[1]
                current_b, current_t, current_nh = parse_config_name(current_config)
            elif current_b is not None and current_t is not None:
                # Metric data row
                range_name = row[0]
                metric_name = row[1]
                value = row[2] if len(row) > 2 else ""
                
                data[(range_name, current_b, current_t, current_nh)][metric_name] = value
                all_metrics[metric_name] = True
                range_order[range_name] = True
    
    return data, all_metrics, range_order

def merge_log_data(data, timing_data, all_metrics):
    """
    Merge timing data from log file into the main data dictionary.
    """
    for key, timing_info in timing_data.items():
        if key not in data:
            data[key] = {}
        
        if 'wall_clock_time' in timing_info:
            data[key]['wall_clock_time_us'] = str(timing_info['wall_clock_time'])
            all_metrics['wall_clock_time_us'] = True
        
        if 'kernel_count' in timing_info:
            data[key]['kernel_count'] = str(timing_info['kernel_count'])
            all_metrics['kernel_count'] = True

def write_output(data, all_metrics, range_order, output_file):
    """Write the reorganized data to output CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Get metrics in order of first appearance
        metrics = list(all_metrics.keys())
        ranges = list(range_order.keys())
        
        # Write column headers
        header = ['Range', 'B', 'T', 'NH'] + metrics
        writer.writerow(header)
        
        # Get all keys and sort by (range_order, B, T, NH)
        all_keys = sorted(data.keys(), key=lambda x: (ranges.index(x[0]) if x[0] in ranges else len(ranges), x[1], x[2], x[3] if x[3] is not None else 12))
        
        # Write data rows
        for range_name, b, t, nh in all_keys:
            row = [range_name, b, t, nh if nh is not None else 12]
            for metric in metrics:
                value = data[(range_name, b, t, nh)].get(metric, '')
                row.append(value)
            writer.writerow(row)
    
    print(f"Output written to {output_file}")

def write_kernel_configs(kernel_configs, output_file):
    """Write kernel grid and block configurations to a text file."""
    with open(output_file, 'w') as f:
        # Group by range name
        range_dict = defaultdict(lambda: defaultdict(list))
        for (range_name, b, t, nh), configs in kernel_configs.items():
            range_dict[range_name][(b, t, nh)] = configs
        
        for range_name in sorted(range_dict.keys()):
            f.write(f"Range name: {range_name}\n")
            
            # Get all (b, t, nh) configs for this range
            bt_configs = range_dict[range_name]
            for (b, t, nh) in sorted(bt_configs.keys()):
                configs = bt_configs[(b, t, nh)]
                if configs:
                    nh_val = nh if nh is not None else 12
                    f.write(f"  B={b}, T={t}, NH={nh_val}:\n")
                    f.write(f"    Kernel launched: ")
                    kernel_strs = []
                    for grid, block in configs:
                        kernel_strs.append(f"[<<<{grid[0]}, {grid[1]}, {grid[2]}>>, <<<{block[0]}, {block[1]}, {block[2]}>>>]")
                    f.write(", ".join(kernel_strs))
                    f.write("\n")
            
            f.write("\n")
    
    print(f"Kernel configurations written to {output_file}")

def main():
    # Define file paths here - modify these variables as needed
    log_file = 'logs_part2.txt'
    input_file = './output/result_part2.csv'
    output_file = './output/reorganized_metrics_part2.csv'
    kernel_config_file = './output/kernel_configs.txt'
    
    print(f"Parsing log file: {log_file}...")
    timing_data, kernel_configs = parse_log_file(log_file)
    print(f"Found {len(timing_data)} range timing entries and {len(kernel_configs)} kernel configurations")
    
    print(f"Parsing CSV file: {input_file}...")
    data, all_metrics, range_order = parse_csv(input_file)
    
    # Merge log data into CSV data
    print("Merging log data with CSV data...")
    merge_log_data(data, timing_data, all_metrics)
    
    num_ranges = len(range_order)
    num_metrics = len(all_metrics)
    num_rows = len(data)
    
    print(f"Found {num_ranges} ranges, {num_metrics} metrics, {num_rows} total rows")
    
    write_output(data, all_metrics, range_order, output_file)
    write_kernel_configs(kernel_configs, kernel_config_file)

if __name__ == '__main__':
    main()