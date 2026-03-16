import os
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import STORAGE, MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AlphaModel")

def prepare_data(df: pd.DataFrame, predict_steps: int):
    """
    Creates feature matrix X and target vector y.
    Target y is the future return N steps ahead.
    """
    # Create the label: future return N steps ahead
    # Since 'returns' is already computed, we can also compute future return relative to current price
    # Simple approach: shift the returns back by N steps
    # We want to predict the cumulative return from t to t+N
    
    df['future_return'] = df['mid_price'].shift(-predict_steps) / df['mid_price'] - 1.0
    
    # Drop NA at the end
    df = df.dropna().copy()
    
    # Select features
    feature_cols = ['spread', 'depth_imbalance', 'returns', 'volatility_10', 'volatility_50', 'ofi']
    
    X = df[feature_cols].values
    y = df['future_return'].values
    timestamps = df['timestamp'].values
    prices = df['mid_price'].values
    
    return X, y, timestamps, prices

def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Model 1: Ridge Regression
    logger.info("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    # Model 2: XGBoost
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Evaluation
    for name, y_pred in [("Ridge", y_pred_ridge), ("XGBoost", y_pred_xgb)]:
        mse = mean_squared_error(y_test, y_pred)
        # Using numpy to calculate pearson correlation
        corr = np.corrcoef(y_test, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0
        logger.info(f"{name} Results -> MSE: {mse:.8f}, Correlation: {corr:.4f}")
        
    return xgb_model, y_pred_xgb

def run_alpha_model():
    logger.info("Starting Alpha Model Pipeline...")
    features_path = os.path.join(STORAGE.PROCESSED_DIR, "features.parquet")
    
    if not os.path.exists(features_path):
        logger.error(f"Features file not found at {features_path}. Run feature engineering first.")
        return
        
    df = pd.read_parquet(features_path)
    if len(df) < 100:
        logger.error("Not enough data to train the model.")
        return
        
    X, y, timestamps, prices = prepare_data(df, MODEL.PREDICT_STEPS)
    
    # Time-series Split
    split_idx = int(len(X) * (1 - MODEL.TEST_SPLIT))
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    prices_test = prices[split_idx:]
    timestamps_test = timestamps[split_idx:]
    
    # Train and evaluate
    model, y_pred_test = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Save predictions alongside actuals back to a parquet for the backtester
    predictions_df = pd.DataFrame({
        'timestamp': timestamps_test,
        'mid_price': prices_test,
        'actual_return': y_test,
        'predicted_return': y_pred_test
    })
    
    pred_path = os.path.join(STORAGE.PROCESSED_DIR, "predictions.parquet")
    predictions_df.to_parquet(pred_path, index=False)
    logger.info(f"Saved test set predictions to {pred_path}")

if __name__ == "__main__":
    run_alpha_model()
