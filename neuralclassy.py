import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mplfinance as mpf



ticker = "ZC=F"
df = yf.download(ticker, start="2020-01-01", end="2026-02-24")
df.dropna(inplace=True)

# FIX 1: Flatten MultiIndex columns
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# FIX 2: Ensure Volume is a 1D Series
if isinstance(df["Volume"], pd.DataFrame):
    df["Volume"] = df["Volume"].iloc[:, 0]
df["Volume"] = df["Volume"].astype(float)


# 3. Feature Engineering
df["return"] = df["Close"].pct_change()
df["volatility"] = df["return"].rolling(10).std()
df["spread_proxy"] = (df["High"] - df["Low"]) / df["Close"]
df["depth_proxy"] = df["Volume"].rolling(5).mean()

# Raw Amihud
df["amihud_raw"] = (df["return"].abs()) / (df["Close"] * df["Volume"])

# Smoothed Amihud (5-day rolling mean)
df["amihud"] = df["amihud_raw"].rolling(5).mean()


# 4. Clean before labeling

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()


# 5. Create 3 liquidity regimes with smoothed regimes

df["label"] = pd.qcut(df["amihud"], q=3, labels=[0, 1, 2]).astype(int)


# 6. SHIFT LABELS FOR 7-DAY FORECASTING
df["label_next_week"] = df["label"].shift(-7)


# 7. FINAL CLEANING (AFTER SHIFTING)
features = ["spread_proxy", "Volume", "volatility", "depth_proxy", "amihud"]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + ["label_next_week"])

print(df.head())
window_df = df.last("365D")

# 8. Prepare features + target
X = df[features].values
y = df["label_next_week"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)


# 9. train test split
split_idx = int(len(df) * 0.8)

X_train = X_tensor[:split_idx]
X_test = X_tensor[split_idx:]
y_train = y_tensor[:split_idx]
y_test = y_tensor[split_idx:]

dates_test = window_df.index
prices_test = window_df["Close"]


# Model
class LiquidityClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.layers(x)


model = LiquidityClassifier(input_dim=len(features))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()

print(f"\n7-Day Ahead Forecast Accuracy: {accuracy:.2%}")


pred_np = predictions.numpy()

# Use the last 1-year window for plotting
window_df = df.last("365D")

# Align predictions to the same window
pred_np = pred_np[-len(window_df):]

# Center day of the test-set plot
center_date = dates_test[int(len(dates_test) / 2)]
print("Center of plot:", center_date)
center_values = df.loc[center_date, ["Close", "High", "Volume", "label_next_week"]]

print("\nValues for center day:")
print("Close:", center_values["Close"])
print("High:", center_values["High"])
print("Volume:", center_values["Volume"])
for date, pred in zip(dates_test, pred_np):
    if(pred==0):
        regime='High Liquidity'
    elif(pred==1):
        regime='Moderate Liquidity'
    elif(pred==2):
        regime='Low Liquidity'
    else:
        regime='error or somethin idk what happened'

print("Regime:", regime)

plt.figure(figsize=(16, 8))
plt.plot(dates_test, prices_test, color="black", linewidth=3)

# THICKER REGIME BARS (1-day width)
for date, pred in zip(dates_test, pred_np):
    if pred == 0:
        plt.axvspan(date, date + pd.Timedelta(days=1), color="green", alpha=0.25)
    elif pred == 2:
        plt.axvspan(date, date + pd.Timedelta(days=1), color="red", alpha=0.25)
plt.axvspan(center_date, center_date + pd.Timedelta(days=1), color='blue', alpha=0.25)
plt.title("7-Day Ahead Liquidity Forecast (Smoothed Amihud Regimes)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("liquidity_plot.png", bbox_inches="tight")
plt.show()


# Volatility Plot
plt.figure(figsize=(16, 6))
plt.plot(window_df.index, window_df["volatility"], color="purple", linewidth=3)
plt.title(f"{ticker} Rolling Volatility (10-Day Window)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("volatility_plot.png", bbox_inches="tight")
plt.show()

ticker = "ZC=F"
dflongshort = yf.download(ticker, start="2024-01-01", end="2026-02-24", auto_adjust=True)


if isinstance(dflongshort.columns, pd.MultiIndex):
    dflongshort.columns = dflongshort.columns.get_level_values(0)
dflongshort.dropna(inplace=True)


price_range = dflongshort['High'] - dflongshort['Low']
dflongshort['buy_ratio'] = np.where(price_range > 0, (dflongshort['Close'] - dflongshort['Low']) / price_range, 0.5)
dflongshort['vol_long'] = dflongshort['Volume'] * dflongshort['buy_ratio']
dflongshort['vol_short'] = dflongshort['Volume'] * (1 - dflongshort['buy_ratio'])
plot_df = dflongshort.tail(30)


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


ticker = "ZC=F"
start_date = "2024-01-01" 
end_date = "2026-02-24" 
dfcandlestick = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)


if isinstance(dfcandlestick.columns, pd.MultiIndex):
    dfcandlestick.columns = dfcandlestick.columns.get_level_values(0)

dfcandlestick.dropna(inplace=True)
plot_df = dfcandlestick.tail(30)

mpf.plot(plot_df, 
         type='candle',         
         style='yahoo',         
         title=f"{ticker} Candlestick Chart (2026)",
         ylabel='Price ($)',           
         mav=(5, 20),           # Optional: Adds 5-day and 20-day moving averages
         tight_layout=True,
         figsize=(16, 8))
