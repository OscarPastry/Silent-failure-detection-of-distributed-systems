import pandas as pd
import numpy as np
import ast
import re
import argparse
import sys

def parse_dict_string(s):
    if pd.isna(s) or s == '[]':
        return np.nan, np.nan
    try:
        s_fixed = s.replace('None', '0.0')
        d = ast.literal_eval(s_fixed)
        if isinstance(d, dict):
            return float(d.get('cpus', np.nan)), float(d.get('memory', np.nan))
    except Exception as e:
        pass
    return np.nan, np.nan

def parse_array_string(s):
    if pd.isna(s) or s == '[]' or s == '':
        return np.nan, np.nan, np.nan
    try:
        s_clean = s.replace('[', '').replace(']', '').replace('\n', ' ')
        vals = [float(x) for x in s_clean.split() if x.strip() != '']
        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        return float(np.mean(vals)), float(np.max(vals)), float(np.std(vals))
    except Exception as e:
        return np.nan, np.nan, np.nan

def run_preprocessing(input_path, output_path, nrows=None):
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path, nrows=nrows)

    print(f"Original shape: {df.shape}")

    # Process dict columns
    dict_columns = ['resource_request', 'average_usage', 'maximum_usage', 'random_sample_usage']
    for col in dict_columns:
        print(f"Processing dict column {col}...")
        parsed = df[col].apply(parse_dict_string)
        df[f'{col}_cpus'] = [x[0] for x in parsed]
        df[f'{col}_memory'] = [x[1] for x in parsed]

    # Process array columns
    array_columns = ['cpu_usage_distribution', 'tail_cpu_usage_distribution']
    for col in array_columns:
        print(f"Processing array column {col}...")
        parsed = df[col].apply(parse_array_string)
        df[f'{col}_mean'] = [x[0] for x in parsed]
        df[f'{col}_max'] = [x[1] for x in parsed]
        df[f'{col}_std'] = [x[2] for x in parsed]

    # Convert features to numeric where possible
    numeric_columns = [
        'time', 'instance_events_type', 'scheduling_class', 'collection_type', 
        'priority', 'instance_index', 'start_time', 'end_time', 
        'assigned_memory', 'page_cache_memory', 'cycles_per_instruction', 
        'memory_accesses_per_instruction', 'sample_rate', 'failed', 'cluster', 'vertical_scaling', 'scheduler'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop non-feature columns
    cols_to_drop = dict_columns + array_columns + [
         'alloc_collection_id', 'collection_name', 'collection_logical_name', 
         'constraint', 'start_after_collection_ids', 'event', 'Unnamed: 0'
    ]
    db_columns_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=db_columns_to_drop, inplace=True)
    
    # Drop rows where target `failed` is nan
    if 'failed' in df.columns:
        df = df.dropna(subset=['failed'])

    # Fill NaNs in feature columns 
    # For a baseline model we can fill 0
    df = df.fillna(0)

    print(f"Processed shape: {df.shape}")
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='borg_traces_data.csv')
    parser.add_argument('--output', type=str, default='processed_traces.parquet')
    parser.add_argument('--nrows', type=int, default=None)
    args = parser.parse_args()
    
    run_preprocessing(args.input, args.output, args.nrows)
