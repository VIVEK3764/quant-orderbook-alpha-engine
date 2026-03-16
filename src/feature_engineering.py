import os
import glob
import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import STORAGE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureEngineering")

def load_latest_data(directory: str) -> pd.DataFrame:
    search_pattern = os.path.join(directory, "**", "*.parquet")
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        return pd.DataFrame()
    
    # Load all files, sort by timestamp
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    
    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate microstructure features from raw orderbook data.
    """
    logger.info(f"Engineering features for {len(df)} records...")
    
    # Mid price
    df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2.0
    
    # Spread
    df['spread'] = df['ask_price'] - df['bid_price']
    
    # Depth Imbalance
    df['depth_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # Returns (pct_change of mid_price)
    df['returns'] = df['mid_price'].pct_change()
    
    # Rolling volatility (e.g., 10-period and 50-period)
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_50'] = df['returns'].rolling(window=50).std()
    
    # Order flow imbalance (simplified as diff in depth imbalance)
    df['ofi'] = df['depth_imbalance'].diff()
    
    # Drop NaNs that are generated from rolling windows and pct_change
    df = df.dropna().reset_index(drop=True)
    
    logger.info(f"Engineered dataset has {len(df)} records.")
    return df

def run_feature_engineering():
    logger.info("Starting feature engineering pipeline...")
    
    ob_df = load_latest_data(STORAGE.RAW_ORDERBOOK_DIR)
    if ob_df.empty:
        logger.warning("No orderbook data found. Cannot run feature engineering.")
        return
        
    features_df = create_features(ob_df)
    
    os.makedirs(STORAGE.PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(STORAGE.PROCESSED_DIR, "features.parquet")
    
    # Save engineered features
    features_df.to_parquet(save_path, index=False)
    logger.info(f"Saved engineered features to {save_path}")

if __name__ == "__main__":
    run_feature_engineering()
