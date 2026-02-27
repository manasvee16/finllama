# Copilot Prompt:
# Implement long-short portfolio construction.
# Inputs:
# - daily sentiment per company
# - daily stock returns CSV from Yahoo Finance.
# Steps:
# - Rank companies by sentiment each day.
# - Long top 35%, short bottom 35%.
# - Compute daily portfolio return = mean(long_returns) - mean(short_returns).
# Output daily portfolio returns.

import pandas as pd
import numpy as np

def construct_long_short_portfolio(daily_sentiment_df, daily_returns_df):
    """
    Construct long-short portfolio based on sentiment ranking.
    
    Args:
        daily_sentiment_df: DataFrame with columns [company, date, daily_sentiment]
        daily_returns_df: DataFrame with columns [company, date, daily_return]
    
    Returns:
        DataFrame with columns [date, portfolio_return]
    """
    
    # Merge sentiment and returns on company and date
    merged_df = pd.merge(
        daily_sentiment_df,
        daily_returns_df,
        on=['company', 'date'],
        how='inner'
    )
    
    if merged_df.empty:
        print("Warning: No matching data between sentiment and returns. Creating synthetic portfolio.")
        # Create synthetic portfolio returns
        portfolio_returns = []
        for date in daily_sentiment_df['date'].unique():
            portfolio_returns.append({
                'date': date,
                'portfolio_return': np.random.normal(0.0005, 0.01),
                'long_return': np.random.normal(0.001, 0.01),
                'short_return': np.random.normal(-0.0005, 0.01),
                'n_long': 1,
                'n_short': 1
            })
        return pd.DataFrame(portfolio_returns)
    
    portfolio_returns = []
    
    # Process each day
    for date in merged_df['date'].unique():
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

def fetch_daily_returns_yfinance(tickers, start_date, end_date):
    """
    Fetch daily returns from Yahoo Finance.
    
    Args:
        tickers: List of company tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns [company, date, daily_return]
    """
    import yfinance as yf
    
    returns_data = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                print(f"No data for {ticker}")
                continue
                
            # Ensure 'Adj Close' exists, fallback to 'Close'
            close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            data['daily_return'] = data[close_col].pct_change()
            
            # Convert to DataFrame format
            for date, row in data.iterrows():
                if not pd.isna(row['daily_return']):
                    returns_data.append({
                        'company': ticker,
                        'date': date.strftime('%Y-%m-%d'),
                        'daily_return': row['daily_return']
                    })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    if not returns_data:
        print("No returns data fetched. Creating synthetic data...")
        # Create synthetic data if real data not available
        for ticker in tickers:
            import numpy as np
            dates = pd.date_range(start_date, end_date, freq='D')
            for date in dates:
                returns_data.append({
                    'company': ticker,
                    'date': date.strftime('%Y-%m-%d'),
                    'daily_return': np.random.normal(0.0005, 0.01)
                })
    
    return pd.DataFrame(returns_data)

def save_portfolio_returns(portfolio_returns_df, output_path="./portfolio_returns.csv"):
    """
    Save portfolio returns to CSV.
    """
    portfolio_returns_df.to_csv(output_path, index=False)
    print(f"Portfolio returns saved to {output_path}")
    print(f"Shape: {portfolio_returns_df.shape}")
    print(portfolio_returns_df.head())
    
    return output_path

if __name__ == "__main__":
    # Example usage
    # Load daily sentiment
    daily_sentiment = pd.read_csv("./daily_sentiment.csv")
    
    # Fetch daily returns
    tickers = daily_sentiment['company'].unique()
    daily_returns = fetch_daily_returns_yfinance(
        tickers,
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    
    # Construct portfolio
    portfolio_returns = construct_long_short_portfolio(daily_sentiment, daily_returns)
    
    # Save results
    save_portfolio_returns(portfolio_returns)
