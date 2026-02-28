import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. ACTIVE ROLL SETUP
f_ticker, n_ticker = "ZCK26.CBT", "ZCN26.CBT" # May vs July 2026
main_ticker = "ZC=F"

print(f"Analyzing {f_ticker} -> {n_ticker} transition velocity...")
df_f = yf.download(f_ticker, period="1y", multi_level_index=False).dropna()
df_n = yf.download(n_ticker, period="1y", multi_level_index=False).dropna()
df_main = yf.download(main_ticker, period="5y", multi_level_index=False).dropna()

# 2. FILTER FOR ACTIVE ROLL PERIOD ONLY
roll_df = pd.DataFrame({'F_Vol': df_f['Volume'], 'N_Vol': df_n['Volume']}).dropna()
# Zoom in: Start when Next contract has at least 2% of the combined volume
roll_df['N_Share'] = (roll_df['N_Vol'] / (roll_df['F_Vol'] + roll_df['N_Vol'])) * 100
roll_df = roll_df[roll_df['N_Share'] > 2] 

# 3. CALCULATE ROLL VELOCITY & PROJECT TARGET DATE
# Velocity = Average daily change in % Share over the last 5 sessions
roll_df['Velocity'] = roll_df['N_Share'].diff().rolling(5).mean()
current_velocity = roll_df['Velocity'].iloc[-1]
current_share = roll_df['N_Share'].iloc[-1]

if current_velocity > 0:
    days_to_crossover = (50 - current_share) / current_velocity
    target_date = datetime.now() + timedelta(days=int(days_to_crossover))
    date_str = target_date.strftime('%Y-%m-%d')
else:
    date_str = "Stalled / Insufficient Data"

# 4. FEATURE ENGINEERING & MODEL (With Regularization)
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

# 5. LIVE INFERENCE
model.eval()
live_row = prepare_features(df_f).iloc[-1:]
with torch.no_grad():
    live_X = torch.tensor(scaler.transform(live_row[features].values), dtype=torch.float32)
    probs = torch.softmax(model(live_X), dim=1)
    conf, pred = torch.max(probs, dim=1)

# 6. DASHBOARD VISUALS
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Graph 1: Zoomed Roll Progress & Target Date Marker
ax1.plot(roll_df.index, roll_df['N_Share'], color='red', linewidth=3, label='Next Contract Share %')
ax1.axhline(50, color='black', linestyle='--', alpha=0.5, label='Crossover Target (50%)')
ax1.set_title(f"ROLL PROGRESS: {f_ticker} -> {n_ticker}", fontweight='bold')
ax1.set_ylabel("% Liquidity Shifted")
ax1.legend()

# Graph 2: Roll Velocity (Acceleration of the shift)
ax2.fill_between(roll_df.index, roll_df['Velocity'], color='blue', alpha=0.3)
ax2.plot(roll_df.index, roll_df['Velocity'], color='blue', label='Velocity (%/Day)')
ax2.axhline(0, color='black', alpha=0.5)
ax2.set_title("VOLUME SHIFT VELOCITY (Speed of the Transition)", fontweight='bold')
ax2.legend()

# DYNAMIC OVERLAY PANEL
plt.figtext(0.5, 0.94, f"ESTIMATED ROLL DATE: {date_str}", ha="center", fontsize=16, fontweight='bold', color='red', bbox=dict(facecolor='white', edgecolor='red', pad=5))
plt.figtext(0.5, 0.02, f"Liquidity Status: {'STABLE' if conf.item() > 0.7 else 'VOLATILE'} | Confidence: {conf.item():.1%}", ha="center", fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.show()

print(f"\nANALYTICS SUMMARY:")
print(f"Current Velocity: {current_velocity:.2f}% shift per day")
print(f"Target Roll Date: {date_str}")
