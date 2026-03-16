import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import pandas as pd

from src.config import STORAGE

def validate_data(directory: str, name: str):
    """Loads the most recent parquet file in a directory structure and prints summary stats."""
    
    # Find all parquet files recursively
    search_pattern = os.path.join(directory, "**", "*.parquet")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"[{name}] No parquet files found in {directory}.")
        return

    # Get the latest file
    latest_file = max(files, key=os.path.getctime)
    
    print(f"\n{'='*50}")
    print(f"[{name}] Latest File: {latest_file}")
    print(f"{'='*50}")
    
    try:
        # Load the parquet file into a pandas dataframe
        df = pd.read_parquet(latest_file)
        
        print("\n--- SCHEMA ---")
        print(df.dtypes)
        
        print("\n--- ROW COUNT ---")
        print(f"Total Rows: {len(df)}")
        
        print("\n--- SUMMARY STATISTICS ---")
        print(df.describe())
        
        print("\n--- HEAD (First 5 Rows) ---")
        print(df.head())
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    validate_data(STORAGE.RAW_ORDERBOOK_DIR, "ORDERBOOK")
    validate_data(STORAGE.RAW_TRADES_DIR, "TRADES")
