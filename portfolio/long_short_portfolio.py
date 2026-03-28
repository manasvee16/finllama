import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime

def construct_long_short_portfolio(daily_sentiment_df, daily_returns_df):
    """
    Construct long-short portfolio based on sentiment ranking.
    
    Args:
        daily_sentiment_df: DataFrame with columns [ticker, date, daily_sentiment]
        daily_returns_df: DataFrame with columns [ticker, date, daily_return]
    
    Returns:
        DataFrame with columns [date, portfolio_return, long_return, short_return, n_long, n_short]
    """
    
    # Rename columns if using 'company' instead of 'ticker'
    if 'company' in daily_sentiment_df.columns:
        daily_sentiment_df = daily_sentiment_df.rename(columns={'company': 'ticker'})
    if 'company' in daily_returns_df.columns:
        daily_returns_df = daily_returns_df.rename(columns={'company': 'ticker'})
    
    # Merge sentiment and returns on ticker and date
    merged_df = pd.merge(
        daily_sentiment_df,
        daily_returns_df,
        on=['ticker', 'date'],
        how='inner'
    )
    
    if merged_df.empty:
        raise ValueError("No valid data available for portfolio backtesting")
    
    portfolio_returns = []
    
    # Process each day
    for date in sorted(merged_df['date'].unique()):
        day_data = merged_df[merged_df['date'] == date].copy()
        
        if len(day_data) < 2:
            continue  # Skip days with too few companies
        
        # Rank companies by sentiment (high to low)
        day_data = day_data.sort_values('daily_sentiment', ascending=False).reset_index(drop=True)
        
        # Calculate 35% cutoff
        n_companies = len(day_data)
        cutoff_idx = max(1, int(np.ceil(n_companies * 0.35)))
        
        # Long positions: top 35% by sentiment
        long_positions = day_data.iloc[:cutoff_idx]
        long_return = long_positions['daily_return'].mean()
        
        # Short positions: bottom 35% by sentiment
        short_positions = day_data.iloc[-cutoff_idx:]
        short_return = short_positions['daily_return'].mean()
        
        # Portfolio return = long - short
        portfolio_return = long_return - short_return
        
        portfolio_returns.append({
            'date': date,
            'portfolio_return': portfolio_return,
            'long_return': long_return,
            'short_return': short_return,
            'n_long': len(long_positions),
            'n_short': len(short_positions)
        })
    
    return pd.DataFrame(portfolio_returns)

def compute_portfolio_metrics(portfolio_returns_df, risk_free_rate=0.02):
    """
    Compute portfolio performance metrics.
    
    Metrics computed:
    - Daily return: Daily portfolio returns
    - Cumulative return: Compound return from start to end
    - Annualized return: Daily return annualized (252 trading days)
    - Volatility: Annualized standard deviation of returns
    - Sharpe ratio: (Annualized return - risk_free_rate) / Annualized volatility
    
    Args:
        portfolio_returns_df: DataFrame with columns [date, portfolio_return]
        risk_free_rate: Annual risk-free rate (default: 0.02 = 2%)
    
    Returns:
        Dictionary with performance metrics
    """
    
    if portfolio_returns_df.empty:
        print("[WARNING] Empty portfolio returns dataframe")
        return {
            'total_days': 0,
            'daily_return': 0,
            'cumulative_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0
        }
    
    returns = portfolio_returns_df['portfolio_return'].values
    
    # Daily statistics
    daily_return = returns.mean()
    daily_volatility = returns.std()
    
    # Cumulative return: (1 + r1) * (1 + r2) * ... - 1
    cumulative_return = np.prod(1 + returns) - 1
    
    # Annualized return and volatility (252 trading days per year)
    annualized_return = (1 + daily_return) ** 252 - 1
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    # Sharpe ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
    
    metrics = {
        'total_days': len(portfolio_returns_df),
        'daily_return': daily_return,
        'daily_volatility': daily_volatility,
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'return_start_date': portfolio_returns_df['date'].iloc[0],
        'return_end_date': portfolio_returns_df['date'].iloc[-1]
    }
    
    return metrics

def print_performance_metrics(metrics):
    """
    Print portfolio performance metrics in a formatted way.
    """
    print("\n" + "="*80)
    print("PORTFOLIO PERFORMANCE METRICS")
    print("="*80)
    print(f"Period: {metrics['return_start_date']} to {metrics['return_end_date']}")
    print(f"Trading days: {metrics['total_days']:,}")
    print("\n[Daily Returns]")
    print(f"  Mean daily return: {metrics['daily_return']:.6f} ({metrics['daily_return']*100:.4f}%)")
    print(f"  Daily volatility: {metrics['daily_volatility']:.6f} ({metrics['daily_volatility']*100:.4f}%)")
    print("\n[Annual Returns]")
    print(f"  Annualized return: {metrics['annualized_return']:.6f} ({metrics['annualized_return']*100:.4f}%)")
    print(f"  Annualized volatility: {metrics['volatility']:.6f} ({metrics['volatility']*100:.4f}%)")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    print("\n[Cumulative Return]")
    print(f"  Total return: {metrics['cumulative_return']:.6f} ({metrics['cumulative_return']*100:.4f}%)")
    print("="*80 + "\n")

def fetch_daily_returns_yfinance(tickers, start_date, end_date):
    """
    Fetch daily returns from Yahoo Finance.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns [ticker, date, daily_return]
    """
    
    returns_data = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                print(f"  No data for {ticker}")
                continue
                
            # Ensure 'Adj Close' exists, fallback to 'Close'
            close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            data['daily_return'] = data[close_col].pct_change()
            
            # Convert to DataFrame format
            for date, row in data.iterrows():
                if not pd.isna(row['daily_return']):
                    returns_data.append({
                        'ticker': ticker,
                        'date': date.strftime('%Y-%m-%d'),
                        'daily_return': row['daily_return']
                    })
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
    
    if not returns_data:
        raise ValueError("No valid data available for portfolio backtesting")

    return pd.DataFrame(returns_data)

def save_portfolio_returns(portfolio_returns_df, output_path="./results/portfolio_returns.csv"):
    """
    Save portfolio returns to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    portfolio_returns_df.to_csv(output_path, index=False)
    print(f"Portfolio returns saved to {output_path}")
    print(f"  Shape: {portfolio_returns_df.shape}")
    print(f"  Date range: {portfolio_returns_df['date'].min()} to {portfolio_returns_df['date'].max()}")
    
    return output_path

def save_performance_metrics(metrics, output_path="./results/portfolio_metrics.csv"):
    """
    Save portfolio performance metrics to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)
    print(f"Performance metrics saved to {output_path}")
    
    return output_path

def run_long_short_portfolio_strategy(
    sentiment_csv="./results/daily_sentiment.csv",
    output_returns_csv="./results/portfolio_returns.csv",
    output_metrics_csv="./results/portfolio_metrics.csv",
    start_date="2023-01-01",
    end_date="2024-12-31"
):
    """
    Run complete long-short portfolio strategy.
    
    Steps:
    1. Load daily sentiment scores
    2. Fetch historical prices from Yahoo Finance
    3. Construct long-short portfolio (rank by sentiment, long top 35%, short bottom 35%)
    4. Compute portfolio daily returns
    5. Calculate performance metrics (return, volatility, Sharpe ratio)
    6. Save results
    
    Args:
        sentiment_csv: Path to daily_sentiment.csv
        output_returns_csv: Path to save portfolio returns
        output_metrics_csv: Path to save performance metrics
        start_date: Start date for price data (YYYY-MM-DD)
        end_date: End date for price data (YYYY-MM-DD)
    
    Returns:
        Tuple of (portfolio_returns_df, metrics_dict)
    """
    print("\n" + "="*80)
    print("LONG-SHORT PORTFOLIO STRATEGY")
    print("="*80 + "\n")
    
    # Step 1: Load daily sentiment
    print("[Step 1] Loading daily sentiment scores...")
    if not os.path.exists(sentiment_csv):
        print(f"[ERROR] Sentiment file not found: {sentiment_csv}")
        return None, None
    
    daily_sentiment = pd.read_csv(sentiment_csv)
    print(f"  Loaded {daily_sentiment.shape[0]:,} sentiment records")

    if "ticker" not in daily_sentiment.columns:
        raise ValueError("No valid data available for portfolio backtesting")

    # Validate ticker column contains non-empty tickers.
    tickers_series = daily_sentiment["ticker"].dropna().astype(str).str.strip()
    if tickers_series.empty:
        raise ValueError("No valid data available for portfolio backtesting")

    print(f"  Unique tickers: {daily_sentiment['ticker'].nunique()}")
    print(f"  Date range: {daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}")
    
    # Step 2: Fetch daily returns
    print(f"\n[Step 2] Fetching historical prices from Yahoo Finance...")
    tickers = daily_sentiment['ticker'].unique().tolist()
    print(f"  Fetching {len(tickers)} tickers from {start_date} to {end_date}...")
    daily_returns = fetch_daily_returns_yfinance(tickers, start_date, end_date)
    print(f"  Loaded {daily_returns.shape[0]:,} return records")
    
    # Step 3: Construct portfolio
    print(f"\n[Step 3] Constructing long-short portfolio...")
    portfolio_returns = construct_long_short_portfolio(daily_sentiment, daily_returns)
    print(f"  Portfolio constructed for {portfolio_returns.shape[0]:,} trading days")
    print(f"  Average portfolio return: {portfolio_returns['portfolio_return'].mean():.6f}")
    print(f"  Long positions per day: {portfolio_returns['n_long'].mean():.1f}")
    print(f"  Short positions per day: {portfolio_returns['n_short'].mean():.1f}")
    
    # Step 4: Compute metrics
    print(f"\n[Step 4] Computing performance metrics...")
    metrics = compute_portfolio_metrics(portfolio_returns)
    print_performance_metrics(metrics)
    
    # Step 5: Save results
    print(f"[Step 5] Saving results...")
    save_portfolio_returns(portfolio_returns, output_returns_csv)
    save_performance_metrics(metrics, output_metrics_csv)
    
    return portfolio_returns, metrics

if __name__ == "__main__":
    # Run long-short portfolio strategy
    portfolio_returns, metrics = run_long_short_portfolio_strategy(
        sentiment_csv="./results/daily_sentiment.csv",
        output_returns_csv="./results/portfolio_returns.csv",
        output_metrics_csv="./results/portfolio_metrics.csv",
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    
    if portfolio_returns is not None:
        print("\nSample portfolio returns:")
        print(portfolio_returns.head(10))
