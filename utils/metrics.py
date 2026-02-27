# Copilot Prompt:
# Implement financial performance metrics:
# - cumulative returns
# - annualized return
# - annualized volatility
# - Sharpe ratio
# Use formulas from the FinLlama base paper.

import numpy as np
import pandas as pd

def compute_cumulative_returns(daily_returns):
    """
    Compute cumulative returns from daily returns.
    
    Formula: CR = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    
    Args:
        daily_returns: Array or Series of daily returns
    
    Returns:
        Cumulative return as float
    """
    return np.prod(1 + daily_returns) - 1

def compute_annualized_return(daily_returns, periods_per_year=252):
    """
    Compute annualized return from daily returns.
    
    Formula: Annual Return = (1 + Cumulative Return) ^ (periods_per_year / n_days) - 1
    
    Args:
        daily_returns: Array or Series of daily returns
        periods_per_year: Trading days per year (default: 252)
    
    Returns:
        Annualized return as float
    """
    n_periods = len(daily_returns)
    cumulative_return = compute_cumulative_returns(daily_returns)
    
    if n_periods == 0:
        return 0
    
    annualized_return = (1 + cumulative_return) ** (periods_per_year / n_periods) - 1
    return annualized_return

def compute_annualized_volatility(daily_returns, periods_per_year=252):
    """
    Compute annualized volatility from daily returns.
    
    Formula: Annual Vol = Daily Vol * sqrt(periods_per_year)
    
    Args:
        daily_returns: Array or Series of daily returns
        periods_per_year: Trading days per year (default: 252)
    
    Returns:
        Annualized volatility as float
    """
    daily_volatility = np.std(daily_returns, ddof=1)
    annualized_volatility = daily_volatility * np.sqrt(periods_per_year)
    return annualized_volatility

def compute_sharpe_ratio(daily_returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Compute Sharpe ratio.
    
    Formula: Sharpe Ratio = (Annual Return - Risk-Free Rate) / Annual Volatility
    
    Args:
        daily_returns: Array or Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 0.02 for 2%)
        periods_per_year: Trading days per year (default: 252)
    
    Returns:
        Sharpe ratio as float
    """
    annual_return = compute_annualized_return(daily_returns, periods_per_year)
    annual_vol = compute_annualized_volatility(daily_returns, periods_per_year)
    
    if annual_vol == 0:
        return 0
    
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
    return sharpe_ratio

def compute_maximum_drawdown(daily_returns):
    """
    Compute maximum drawdown.
    
    Args:
        daily_returns: Array or Series of daily returns
    
    Returns:
        Maximum drawdown as float (negative value)
    """
    cumulative = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    return max_drawdown

def compute_all_metrics(daily_returns_df, risk_free_rate=0.02, periods_per_year=252):
    """
    Compute all performance metrics for a portfolio.
    
    Args:
        daily_returns_df: DataFrame with 'date' and 'portfolio_return' columns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading days per year
    
    Returns:
        Dictionary with all metrics
    """
    # Handle empty dataframe
    if daily_returns_df.empty or 'portfolio_return' not in daily_returns_df.columns:
        return {
            'cumulative_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'maximum_drawdown': 0.0,
            'total_days': 0,
            'risk_free_rate': risk_free_rate
        }
    
    daily_returns = daily_returns_df['portfolio_return'].values
    
    if len(daily_returns) == 0:
        return {
            'cumulative_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'maximum_drawdown': 0.0,
            'total_days': 0,
            'risk_free_rate': risk_free_rate
        }
    
    metrics = {
        'cumulative_return': compute_cumulative_returns(daily_returns),
        'annualized_return': compute_annualized_return(daily_returns, periods_per_year),
        'annualized_volatility': compute_annualized_volatility(daily_returns, periods_per_year),
        'sharpe_ratio': compute_sharpe_ratio(daily_returns, risk_free_rate, periods_per_year),
        'maximum_drawdown': compute_maximum_drawdown(daily_returns),
        'total_days': len(daily_returns),
        'risk_free_rate': risk_free_rate
    }
    
    return metrics

def print_metrics(metrics):
    """
    Print metrics in a formatted table.
    """
    print("\n" + "="*50)
    print("PORTFOLIO PERFORMANCE METRICS")
    print("="*50)
    print(f"Cumulative Return:      {metrics['cumulative_return']:.2%}")
    print(f"Annualized Return:      {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility:  {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown:       {metrics['maximum_drawdown']:.2%}")
    print(f"Trading Days:           {metrics['total_days']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Example usage
    sample_returns = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'portfolio_return': np.random.normal(0.0005, 0.01, 100)
    })
    
    metrics = compute_all_metrics(sample_returns)
    print_metrics(metrics)
