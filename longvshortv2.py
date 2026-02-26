import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf


ticker = "ZC=F"
df = yf.download(ticker, start="2024-01-01", end="2026-02-24", auto_adjust=True)


if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)


price_range = df['High'] - df['Low']
df['buy_ratio'] = np.where(price_range > 0, (df['Close'] - df['Low']) / price_range, 0.5)
df['vol_long'] = df['Volume'] * df['buy_ratio']
df['vol_short'] = df['Volume'] * (1 - df['buy_ratio'])
plot_df = df.tail("365D")


ap = [
    mpf.make_addplot(plot_df['vol_long'], type='bar', color='#26a69a', panel=0, ylabel='Volume Blocks'),
    mpf.make_addplot(plot_df['vol_short'], type='bar', color='#ef5350', panel=0, bottom=plot_df['vol_long'])
]


mpf.plot(plot_df, 
         type='line',          
         linecolor='none',     
         addplot=ap,           
         style='charles',      
         title=f"\n{ticker} Volume Building Blocks",
         figsize=(16, 8),
         datetime_format='%b %d',
         xrotation=45,
         tight_layout=True,
         axisoff=False)        
