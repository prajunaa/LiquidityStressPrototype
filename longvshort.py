import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ticker = "ZC=F"
start_date = "2023-01-01"
end_date = "2026-02-24" 
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)


if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)


high = df['High'].values.flatten()
low = df['Low'].values.flatten()
close = df['Close'].values.flatten()
volume = df['Volume'].values.flatten()
price_range = high - low
buy_ratio = np.where(price_range > 0, (close - low) / price_range, 0.5)


df['vol_long'] = volume * buy_ratio
df['vol_short'] = volume * (1 - buy_ratio)
df[['vol_long', 'vol_short']] = df[['vol_long', 'vol_short']].fillna(0)

plot_df = df.tail(45)

plt.figure(figsize=(16, 8))


plt.bar(plot_df.index, plot_df['vol_long'], 
        color='#26a69a', alpha=0.9, label='Long Portion', width=0.7)


plt.bar(plot_df.index, plot_df['vol_short'], 
        bottom=plot_df['vol_long'], # This 'stacks' the block
        color='#ef5350', alpha=0.9, label='Short Portion', width=0.7)


plt.title(f"{ticker} Volume Building Blocks: Long vs Short Split (2026)", fontsize=15, fontweight='bold')
plt.ylabel("Contracts (Total Daily Volume)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(frameon=True, shadow=True, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
