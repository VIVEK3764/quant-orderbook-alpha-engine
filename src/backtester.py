import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import STORAGE
from src.metrics import print_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Backtester")

def run_backtest(df: pd.DataFrame, threshold: float = 0.0001, transaction_cost: float = 0.0002):
    """
    Simple vectorized backtester.
    Rule:
      If pred > threshold -> LONG (+1)
      If pred < -threshold -> SHORT (-1)
      Else -> FLAT (0)
    """
    logger.info(f"Running backtest on {len(df)} periods..")
    
    # 1. Generate signals
    df['signal'] = 0
    df.loc[df['predicted_return'] > threshold, 'signal'] = 1
    df.loc[df['predicted_return'] < -threshold, 'signal'] = -1
    
    # 2. Calculate strategy returns
    # Strategy return is the signal of the *current* period multiplied by the *actual future return*.
    # (Because actual_return was already shifted backwards to align with the prediction time).
    # NOTE: In a true tick-by-tick simulation, we need to be careful about when the trade executes.
    # Here, we assume the trade executes and captures exactly `actual_return`, minus costs.
    
    # Strategy gross returns
    df['strategy_return'] = df['signal'] * df['actual_return']
    
    # Calculate costs (applied when position changes)
    # We approximate slippage/fees by deducting transaction_cost for every unit of position change.
    df['position_change'] = df['signal'].diff().fillna(0).abs()
    df['costs'] = df['position_change'] * transaction_cost
    
    # Strategy net returns
    df['strategy_return_net'] = df['strategy_return'] - df['costs']
    
    # Cumulative Equity Curve
    # Starting with 1.0 (100%)
    df['equity_curve'] = (1 + df['strategy_return_net']).cumprod()
    
    return df

def plot_equity_curve(df: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['equity_curve'], label='Strategy Equity')
    plt.title('Backtest Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    logger.info("Starting Backtesting Engine...")
    pred_path = os.path.join(STORAGE.PROCESSED_DIR, "predictions.parquet")
    
    if not os.path.exists(pred_path):
        logger.error(f"Predictions file not found at {pred_path}. Run alpha model first.")
        return
        
    df = pd.read_parquet(pred_path)
    
    # Define thresholds
    THRESHOLD = 0.00005 # 0.5 bps
    COST = 0.0001 # 1 bps (Binance VIP 0 maker is 0 bps, but 1 bps standard with some fees)
    
    results = run_backtest(df, threshold=THRESHOLD, transaction_cost=COST)
    
    metrics_returns = results['strategy_return_net'].values
    equity_curve = results['equity_curve'].values
    
    print_metrics(metrics_returns, equity_curve)
    
    plot_path = os.path.join(STORAGE.PROCESSED_DIR, "equity_curve.png")
    plot_equity_curve(results, save_path=plot_path)
    logger.info(f"Saved equity curve plot to {plot_path}")

if __name__ == "__main__":
    main()
