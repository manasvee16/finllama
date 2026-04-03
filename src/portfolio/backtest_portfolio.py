import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
# Paths updated for the new folder structure
PREDICTION_FILES = {
    "Base Llama (Untrained)": Path("../../results/predictions/Base_Llama_predictions.csv"),
    "FinLLaMA (LoRA Tuned)": Path("../../results/predictions/FinLLaMA_predictions.csv")
}
OUTPUT_DIR = Path("../../results/figures")
TICKER = "SPY" # Trading the S&P 500 ETF

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates the annualized Sharpe Ratio"""
    if returns.std() == 0:
        return 0
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

def main():
    print("="*60)
    print("PORTFOLIO BACKTESTER: BASE VS TUNED")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    portfolio_data = {}
    min_date = None
    max_date = None

    # 1. Process sentiments into daily trading signals
    for model_name, file_path in PREDICTION_FILES.items():
        if not file_path.exists():
            print(f"Warning: Skipping {model_name} - Could not find {file_path.resolve()}")
            continue
            
        print(f"Processing trading signals for {model_name}...")
        df = pd.read_csv(file_path)
        
        if 'date' not in df.columns:
            print(f"Error: No 'date' column found in {file_path.name}")
            return
            
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.floor('D')
        
        current_min = df['date'].min()
        current_max = df['date'].max()
        if min_date is None or current_min < min_date: min_date = current_min
        if max_date is None or current_max > max_date: max_date = current_max

        sentiment_map = {2: 1, 0: -1, 1: 0}
        df['signal_value'] = df['predicted_sentiment'].map(sentiment_map)
        
        daily_sentiment = df.groupby('date')['signal_value'].mean()
        
        daily_signals = np.where(daily_sentiment > 0.1, 1, 
                        np.where(daily_sentiment < -0.1, -1, 0))
        
        signal_df = pd.DataFrame({'date': daily_sentiment.index, f'{model_name}_Signal': daily_signals})
        signal_df.set_index('date', inplace=True)
        portfolio_data[model_name] = signal_df

    if not portfolio_data:
        print("Error: No data available to backtest.")
        return

    # 2. Fetch Market Data
    fetch_end_date = max_date + pd.Timedelta(days=5)
    print(f"\nFetching market data for {TICKER} from {min_date.date()} to {fetch_end_date.date()}...")
    market_data = yf.download(TICKER, start=min_date, end=fetch_end_date, progress=False)
    
    if isinstance(market_data.columns, pd.MultiIndex):
        market_data.columns = market_data.columns.get_level_values(0)
 
    if market_data.empty:
        print("Error: Failed to fetch market data.")
        return

    market_data['Market_Return'] = market_data['Close'].pct_change()
    market_data.index = market_data.index.tz_localize(None).floor('D')

    # 3. Merge Signals with Market Data
    backtest_df = market_data[['Market_Return']].copy()
    backtest_df['Buy_and_Hold'] = (1 + backtest_df['Market_Return']).cumprod()
    
    metrics = []

    for model_name, signal_df in portfolio_data.items():
        backtest_df = backtest_df.join(signal_df, how='left')
        backtest_df[f'{model_name}_Signal'] = backtest_df[f'{model_name}_Signal'].ffill().fillna(0) 
        backtest_df[f'{model_name}_Position'] = backtest_df[f'{model_name}_Signal'].shift(1)
        
        strategy_col = f'{model_name}_Return'
        backtest_df[strategy_col] = backtest_df[f'{model_name}_Position'] * backtest_df['Market_Return']
        backtest_df[f'{model_name}_Cumulative'] = (1 + backtest_df[strategy_col]).cumprod()
        
        total_return = backtest_df[f'{model_name}_Cumulative'].iloc[-1] - 1
        sharpe = calculate_sharpe_ratio(backtest_df[strategy_col])
        metrics.append({"Model": model_name, "Total Return": f"{total_return*100:.2f}%", "Sharpe Ratio": f"{sharpe:.2f}"})

    # Add Baseline Metrics
    bnh_return = backtest_df['Buy_and_Hold'].iloc[-1] - 1
    bnh_sharpe = calculate_sharpe_ratio(backtest_df['Market_Return'])
    metrics.append({"Model": "S&P 500 (Buy & Hold)", "Total Return": f"{bnh_return*100:.2f}%", "Sharpe Ratio": f"{bnh_sharpe:.2f}"})

    # 4. Print Scoreboard
    metrics_df = pd.DataFrame(metrics)
    print("\n" + "="*40)
    print("FINAL PORTFOLIO PERFORMANCE")
    print("="*40)
    print(metrics_df.to_string(index=False))

    # 5. Plotting
    print("\nGenerating performance chart...")
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # Plot Market Baseline
    plt.plot(backtest_df.index, backtest_df['Buy_and_Hold'], label='S&P 500 (Buy & Hold)', color='black', linestyle='--', linewidth=2)
    
    # Explicit colors: Red for untrained, Green for tuned
    colors = {'Base Llama (Untrained)': '#e74c3c', 'FinLLaMA (LoRA Tuned)': '#2ecc71'}
    
    for model_name in portfolio_data.keys():
        plt.plot(backtest_df.index, backtest_df[f'{model_name}_Cumulative'], label=model_name, color=colors[model_name], linewidth=2)

    plt.title("Long/Short Portfolio Simulation: Financial Sentiment Strategies", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Growth (1.0 = Initial Investment)", fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    
    plot_path = OUTPUT_DIR / "portfolio_backtest_chart.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Chart saved to: {plot_path.resolve()}")
    
    plt.show()

if __name__ == "__main__":
    main()