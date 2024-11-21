"""
Script to merge multiple CSV files from the dataset folder, aligning them by date.
This is designed for building a Solana price prediction model.
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

def parse_date(date_str):
    """Parse various date formats and return a standardized date string."""
    if pd.isna(date_str):
        return None
    
    # Remove quotes if present
    date_str = str(date_str).strip().strip('"').strip("'")
    
    # Try different date formats
    formats = [
        "%Y-%m-%d %H:%M:%S UTC",  # Crypto format: 2020-04-11 00:00:00 UTC
        "%Y-%m-%d",               # ISO format: 2020-04-11
        "%m/%d/%Y",               # US format: 11/21/2025
        "%d/%m/%Y",               # European format: 21/11/2025
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.date()  # Return just the date part
        except ValueError:
            continue
    
    # If all formats fail, try pandas parsing
    try:
        dt = pd.to_datetime(date_str)
        return dt.date()
    except:
        print(f"Warning: Could not parse date: {date_str}")
        return None

def load_and_prepare_csv(file_path, date_column=None):
    """Load a CSV file and prepare it for merging."""
    try:
        # Read CSV, handling different encodings and quote styles
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')
        
        if df.empty:
            print(f"Warning: Empty file {file_path}")
            return None
        
        # Find the date column
        if date_column is None:
            # Common date column names
            date_cols = ['snapped_at', 'Date', 'date', 'DATE', 'timestamp', 'Timestamp']
            date_column = None
            for col in date_cols:
                if col in df.columns:
                    date_column = col
                    break
        
        if date_column is None:
            print(f"Warning: No date column found in {file_path}")
            return None
        
        # Parse dates
        df['date'] = df[date_column].apply(parse_date)
        
        # Remove rows with invalid dates
        df = df[df['date'].notna()].copy()
        
        if df.empty:
            print(f"Warning: No valid dates found in {file_path}")
            return None
        
        # Remove the original date column if it's different from 'date'
        if date_column != 'date':
            df = df.drop(columns=[date_column])
        
        # Add prefix to column names (except date) based on filename
        filename = Path(file_path).stem
        prefix = filename.replace('-', '_').replace(' ', '_')
        
        # Rename columns (except 'date')
        rename_dict = {}
        for col in df.columns:
            if col != 'date':
                # If column already has a prefix, keep it; otherwise add one
                if not col.startswith(prefix):
                    rename_dict[col] = f"{prefix}_{col}"
                else:
                    rename_dict[col] = col
        
        df = df.rename(columns=rename_dict)
        
        # Remove duplicates by date (keep first occurrence)
        df = df.drop_duplicates(subset=['date'], keep='first')
        
        # Clean numeric columns (remove commas, handle K/M suffixes)
        for col in df.columns:
            if col != 'date':
                if df[col].dtype == 'object':
                    # Try to convert to numeric, handling commas and K/M suffixes
                    try:
                        # Remove commas and convert K/M suffixes
                        def clean_number(val):
                            if pd.isna(val):
                                return val
                            val_str = str(val).replace(',', '').strip()
                            if val_str.endswith('K'):
                                return float(val_str[:-1]) * 1000
                            elif val_str.endswith('M'):
                                return float(val_str[:-1]) * 1000000
                            elif val_str.endswith('%'):
                                return float(val_str[:-1])
                            else:
                                return float(val_str)
                        
                        df[col] = df[col].apply(clean_number)
                    except:
                        # If conversion fails, keep as is
                        pass
        
        return df
    
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def merge_datasets(dataset_folder='dataset', output_file='merged_dataset.csv'):
    """Merge all CSV files in the dataset folder by date."""
    
    dataset_path = Path(dataset_folder)
    if not dataset_path.exists():
        print(f"Error: Dataset folder '{dataset_folder}' not found!")
        return None
    
    # Get all CSV files
    csv_files = list(dataset_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in '{dataset_folder}' folder!")
        return None
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Load all datasets
    dataframes = []
    solana_df = None
    solana_file = None
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        df = load_and_prepare_csv(csv_file)
        if df is not None:
            print(f"  Loaded {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")
            
            # Identify Solana dataset (target variable)
            if 'sol' in csv_file.name.lower() or 'solana' in csv_file.name.lower():
                solana_df = df
                solana_file = csv_file.name
                print(f"  *** Identified as Solana dataset (target variable) ***")
            
            dataframes.append(df)
        else:
            print(f"  Failed to load {csv_file.name}")
    
    if not dataframes:
        print("No datasets were successfully loaded!")
        return None
    
    # Find the earliest Solana date to filter out pre-launch data
    if solana_df is not None:
        solana_start_date = solana_df['date'].min()
        print(f"\n{'='*60}")
        print(f"Solana data starts on: {solana_start_date}")
        print(f"Filtering out all data before Solana launch...")
        print(f"{'='*60}")
    else:
        print(f"\nWarning: Solana dataset not found! Proceeding without date filtering.")
        solana_start_date = None
    
    # Start with the first dataframe
    merged_df = dataframes[0].copy()
    
    # Merge all other dataframes
    for i, df in enumerate(dataframes[1:], 1):
        print(f"\nMerging dataset {i+1}/{len(dataframes)-1}...")
        merged_df = pd.merge(merged_df, df, on='date', how='outer', suffixes=('', f'_dup{i}'))
    
    # Sort by date
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    # Remove duplicate columns (if any)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    
    # Reorder columns: date first, Solana columns last (target feature), others in between
    if 'date' in merged_df.columns:
        # Identify Solana-related columns (target features)
        solana_cols = [col for col in merged_df.columns if 'sol' in col.lower() and col != 'date']
        
        # Get all other columns (excluding date and Solana columns)
        other_cols = [col for col in merged_df.columns if col != 'date' and col not in solana_cols]
        
        # Reorder: date first, other features, then Solana (target) at the end
        cols = ['date'] + other_cols + solana_cols
        merged_df = merged_df[cols]
        
        if solana_cols:
            print(f"\nReordered columns: date first, Solana target features last")
            print(f"Solana columns (target): {solana_cols}")
    
    # Filter out data before Solana launch (since we're predicting Solana price)
    if solana_start_date is not None:
        rows_before = len(merged_df)
        merged_df = merged_df[merged_df['date'] >= solana_start_date].copy()
        rows_after = len(merged_df)
        print(f"\nFiltered out {rows_before - rows_after} rows before Solana launch")
        print(f"Remaining rows: {rows_after}")
    
    print(f"\n{'='*60}")
    print(f"Merged dataset created successfully!")
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"{'='*60}")
    
    # Display column names
    print(f"\nColumns in merged dataset:")
    for i, col in enumerate(merged_df.columns, 1):
        non_null_count = merged_df[col].notna().sum()
        print(f"  {i:2d}. {col:40s} ({non_null_count} non-null values)")
    
    # Save to CSV
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged dataset saved to: {output_file}")
    
    return merged_df

if __name__ == "__main__":
    # Merge all datasets
    merged_data = merge_datasets()
    
    if merged_data is not None:
        print("\nFirst few rows of merged dataset:")
        print(merged_data.head(10))
        print("\nLast few rows of merged dataset:")
        print(merged_data.tail(10))

