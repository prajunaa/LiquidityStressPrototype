import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta


# 1. ACTIVE ROLL SETUP (Corn May 2026 -> July 2026)
f_ticker, n_ticker = "ZCK26.CBT", "ZCN26.CBT"
main_ticker = "ZC=F"

print(f"Syncing {f_ticker} -> {n_ticker} for March 2026 Roll...")
df_f = yf.download(f_ticker, period="1y", multi_level_index=False).dropna()
df_n = yf.download(n_ticker, period="1y", multi_level_index=False).dropna()
df_main = yf.download(main_ticker, period="5y", multi_level_index=False).dropna()

# 2. FEATURE ENGINEERING & MODEL
def prepare_features(df_in):
    df = df_in.copy()
    df["return"] = df["Close"].pct_change()
    df["vol"] = df["return"].rolling(10).std()
    df["amihud"] = (df["return"].abs() / (df["Close"] * df["Volume"] / 1_000_000)).rolling(5).mean()
    df["rvol"] = df["Volume"] / df["Volume"].rolling(10).mean()
    return df.dropna()

train_df = prepare_features(df_main)
features = ["vol", "amihud", "rvol"]
t_low, t_high = train_df["amihud"].quantile(0.33), train_df["amihud"].quantile(0.66)
train_df["label"] = train_df["amihud"].apply(lambda x: 0 if x <= t_low else (2 if x > t_high else 1))
train_df["target"] = train_df["label"].shift(-7)
train_data = train_df.dropna(subset=["target"])

scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(train_data[features].values), dtype=torch.float32)
y = torch.tensor(train_data["target"].values, dtype=torch.long)

class LiquidityMax(nn.Module):
    def __init__(self, in_d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 3))
    def forward(self, x): return self.net(x)

model = LiquidityMax(len(features))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for _ in range(200):
    optimizer.zero_grad(); nn.functional.cross_entropy(model(X), y).backward(); optimizer.step()

# 3. LIVE INFERENCE & DYNAMIC VELOCITY
model.eval()
live_row = prepare_features(df_f).iloc[-1:]
with torch.no_grad():
    live_X = torch.tensor(scaler.transform(live_row[features].values), dtype=torch.float32)
    probs = torch.softmax(model(live_X), dim=1)
    conf, pred = torch.max(probs, dim=1)

# Filter 30-day window for roll analysis
roll_df = pd.DataFrame({'F_Vol': df_f['Volume'], 'N_Vol': df_n['Volume']}).dropna().tail(30)
roll_df['N_Share'] = (roll_df['N_Vol'] / (roll_df['F_Vol'] + roll_df['N_Vol'])) * 100
current_share = roll_df['N_Share'].iloc[-1]

# AI-ADJUSTED VELOCITY: If model predicts high volatility (Label 2), we assume the roll slows down.
raw_velocity = roll_df['N_Share'].diff().rolling(5).mean().iloc[-1]
# Penalty: High risk (pred=2) reduces speed by 40%; Stable (pred=0) increases speed by 10%
velocity_adj = 0.6 if pred == 2 else 1.1 if pred == 0 else 1.0
ai_velocity = max(0.1, raw_velocity * velocity_adj) 

# 4. FORECAST GENERATION
days_to_crossover = (50 - current_share) / ai_velocity if ai_velocity > 0 else 0
target_date = datetime.now() + timedelta(days=int(days_to_crossover))

forecast_dates = pd.date_range(start=roll_df.index[-1] + timedelta(days=1), periods=int(days_to_crossover) + 5)
projected_shares = np.clip(current_share + (ai_velocity * np.arange(1, len(forecast_dates) + 1)), 0, 100)

# 5. DASHBOARD VISUALS
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top Graph: Historical Volume
ax1.bar(roll_df.index, roll_df['F_Vol'], label=f'Expiring ({f_ticker})', color='gray', alpha=0.3)
ax1.bar(roll_df.index, roll_df['N_Vol'], label=f'Next ({n_ticker})', color='blue', alpha=0.6)
ax1.set_title("30-DAY MARKET MIGRATION VOLUME", fontweight='bold')
ax1.legend()

# Bottom Graph: The AI Forecast Path
ax2.plot(roll_df.index, roll_df['N_Share'], color='black', label='Actual Share %', linewidth=2)
ax2.plot(forecast_dates, projected_shares, color='red', linestyle='--', label='AI Projected Path')
ax2.axhline(50, color='red', alpha=0.3, label='50% Crossover')
ax2.fill_between(forecast_dates, 50, projected_shares, where=(projected_shares >= 50), color='green', alpha=0.2)

# Crossover Marker
ax2.annotate(f"EST. ROLL: {target_date.strftime('%b %d')}", xy=(target_date, 50), 
             xytext=(target_date, 20), arrowprops=dict(arrowstyle="->", color='red'), color='red', fontweight='bold')

ax2.set_title(f"LIQUIDITY CROSSOVER FORECAST (Speed: {ai_velocity:.1f}% / Day)", fontweight='bold')
ax2.set_ylabel("% Liquidity in Next Contract")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.legend(loc='upper left')

# Bottom Status Panel
status_msg = "STABLE" if pred == 0 else "VOLATILE"
plt.figtext(0.5, 0.02, f"AI LIKELY STATE: {status_msg} | CONFIDENCE: {conf.item():.1%} | TARGET: {target_date.strftime('%Y-%m-%d')}", 
            ha="center", fontsize=11, fontweight='bold', color='white', bbox=dict(facecolor='green' if pred == 0 else 'red', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
