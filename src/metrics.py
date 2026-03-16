import numpy as np

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 365*24*60*60) -> float:
    """Calculate annualized Sharpe ratio. (Assuming inputs are ~1 second or tick level).
    If we assume the periods are 100ms ticks, periods_per_year = 365*24*60*60*10.
    For simplicity, let's just return a standard ratio.
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    
    # Adjust periods_per_year based on data frequency.
    # We'll use a rough generic annualization factor.
    mean_ret = np.mean(returns) - risk_free_rate
    std_ret = np.std(returns)
    
    return (mean_ret / std_ret) * np.sqrt(periods_per_year)

def max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate the maximum drawdown of an equity curve."""
    if len(equity_curve) == 0:
        return 0.0
        
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    return np.max(drawdowns)

def annualized_return(cumulative_return: float, num_periods: int, periods_per_year: int = 365*24*60*60) -> float:
    """Calculate annualized return based on total cumulative return."""
    if num_periods == 0:
        return 0.0
    years = num_periods / periods_per_year
    if years == 0:
        return 0.0
    return (1 + cumulative_return) ** (1 / years) - 1

def volatility(returns: np.ndarray, periods_per_year: int = 365*24*60*60) -> float:
    """Calculate annualized volatility."""
    if len(returns) < 2:
        return 0.0
    return np.std(returns) * np.sqrt(periods_per_year)

def hit_rate(returns: np.ndarray) -> float:
    """Calculate the percentage of profitable trades/periods."""
    if len(returns) == 0:
        return 0.0
    wins = np.sum(returns > 0)
    return wins / len(returns)

def print_metrics(returns: np.ndarray, equity_curve: np.ndarray):
    """Utility to print all standard metrics."""
    # Note: the period annualization factor strictly depends on the interval of the returns.
    # Assuming ~1 second intervals for this example.
    annualization_factor = 365 * 24 * 60 * 60 # 1 second periods
    
    sr = sharpe_ratio(returns, periods_per_year=annualization_factor)
    mdd = max_drawdown(equity_curve)
    vol = volatility(returns, periods_per_year=annualization_factor)
    hr = hit_rate(returns)
    
    total_ret = equity_curve[-1] - 1.0 if len(equity_curve) > 0 else 0.0
    ann_ret = annualized_return(total_ret, len(returns), periods_per_year=annualization_factor)
    
    print("\n--- STRATEGY PERFORMANCE METRICS ---")
    print(f"Total Cumulative Return: {total_ret * 100:.4f}%")
    print(f"Annualized Return:       {ann_ret * 100:.4f}%")
    print(f"Annualized Volatility:   {vol * 100:.4f}%")
    print(f"Sharpe Ratio:            {sr:.4f}")
    print(f"Maximum Drawdown:        {mdd * 100:.4f}%")
    print(f"Hit Rate:                {hr * 100:.2f}%")
    print("------------------------------------\n")
