"""Quick script to reorder columns: date first, Solana columns last (target feature)"""
import pandas as pd

df = pd.read_csv('merged_dataset.csv')
if 'date' in df.columns:
    # Identify Solana-related columns (target features)
    solana_cols = [col for col in df.columns if 'sol' in col.lower() and col != 'date']
    
    # Get all other columns (excluding date and Solana columns)
    other_cols = [col for col in df.columns if col != 'date' and col not in solana_cols]
    
    # Reorder: date first, other features, then Solana (target) at the end
    cols = ['date'] + other_cols + solana_cols
    df = df[cols]
    df.to_csv('merged_dataset.csv', index=False)
    print(f"Columns reordered successfully. Total columns: {len(df.columns)}")
    print(f"First column: {df.columns[0]}")
    print(f"Last columns (Solana target): {list(df.columns[-len(solana_cols):]) if solana_cols else 'None found'}")
else:
    print("Date column not found in dataset")

