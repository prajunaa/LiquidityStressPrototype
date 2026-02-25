import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf  


ticker = "ZC=F"
start_date = "2024-01-01" 
end_date = "2026-02-24" 
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)


if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.dropna(inplace=True)
plot_df = df.tail(45)

mpf.plot(plot_df, 
         type='candle',         
         style='yahoo',         
         title=f"{ticker} Candlestick Chart (2026)",
         ylabel='Price ($)',           
         mav=(5, 20),           # Optional: Adds 5-day and 20-day moving averages
         tight_layout=True,
         figsize=(16, 8))