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
    """Extract B and T values from config name like 'B=4T=64'."""
    match = re.match(r'B=(\d+)T=(\d+)', config_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_csv(input_file):
    """
    Parse the input CSV file.
    Returns a dict: {(range_name, B, T): {metric_name: value}}
    Also returns the ordered list of all metrics.
    """
    data = defaultdict(dict)
    all_metrics = OrderedDict()
    range_order = OrderedDict()
    
    current_config = None
    current_b = None
    current_t = None
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            
            if row[0] == 'Config Name':
                # New configuration
                current_config = row[1]
                current_b, current_t = parse_config_name(current_config)
            elif current_b is not None and current_t is not None:
                # Metric data row
                range_name = row[0]
                metric_name = row[1]
                value = row[2] if len(row) > 2 else ""
                
                data[(range_name, current_b, current_t)][metric_name] = value
                all_metrics[metric_name] = True
                range_order[range_name] = True
    
    return data, all_metrics, range_order

def write_output(data, all_metrics, range_order, output_file):
    """Write the reorganized data to output CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Get metrics in order of first appearance
        metrics = list(all_metrics.keys())
        ranges = list(range_order.keys())
        
        # Write column headers
        header = ['Range', 'B', 'T'] + metrics
        writer.writerow(header)
        
        # Get all keys and sort by (range_order, B, T)
        all_keys = sorted(data.keys(), key=lambda x: (ranges.index(x[0]), x[1], x[2]))
        
        # Write data rows
        for range_name, b, t in all_keys:
            row = [range_name, b, t]
            for metric in metrics:
                value = data[(range_name, b, t)].get(metric, '')
                row.append(value)
            writer.writerow(row)
    
    print(f"Output written to {output_file}")

def main():
    input_file = './output/result.csv'
    output_file = './output/reorganized_metrics.csv'
    
    print(f"Parsing {input_file}...")
    data, all_metrics, range_order = parse_csv(input_file)
    
    num_ranges = len(range_order)
    num_metrics = len(all_metrics)
    num_rows = len(data)
    
    print(f"Found {num_ranges} ranges, {num_metrics} metrics, {num_rows} total rows")
    
    write_output(data, all_metrics, range_order, output_file)

if __name__ == '__main__':
    main()