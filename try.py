import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 1. DATA FETCH
ticker = "ZC=F"
df = yf.download(ticker, period="5y", interval="1d", multi_level_index=False)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)

#2. ENHANCED FEATURES (The "Confidence Boosters") 
df["return"] = df["Close"].pct_change()
df["vol"] = df["return"].rolling(10).std()
df["amihud"] = ((df["return"].abs()) / (df["Close"] * df["Volume"])).rolling(5).mean()

# High-Alpha 1: Price Acceleration (2-day vs 10-day trend)
df["accel"] = df["Close"].pct_change(2) - df["Close"].pct_change(10)

# High-Alpha 2: Volume Conviction (Today vs 5-day mean)
df["vol_convict"] = df["Volume"] / df["Volume"].rolling(5).mean()

# High-Alpha 3: BB Width (Market Tension)
df["std_20"] = df["Close"].rolling(20).std()
df["bb_width"] = (df["std_20"] * 4) / df["Close"].rolling(20).mean()

features = ["Volume", "vol", "amihud", "accel", "vol_convict", "bb_width"]

# LABELING 
hist_amihud = df["amihud"].dropna()
t_low, t_high = hist_amihud.quantile(0.33), hist_amihud.quantile(0.66)
df["label"] = df["amihud"].apply(lambda x: 0 if x <= t_low else (2 if x > t_high else 1))

#4. DATA SPLIT
df_ready = df.dropna(subset=features)
live_row = df_ready.iloc[-1:].copy() 
train_df = df_ready.dropna(subset=["label"]).copy()
train_df["target"] = train_df["label"].shift(-7)
train_data = train_df.dropna(subset=["target"])

#5.MODEL (More capacity for complex patterns)
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(train_data[features].values), dtype=torch.float32)
y_train = torch.tensor(train_data["target"].values, dtype=torch.long)

class LiquidityMax(nn.Module):
    def __init__(self, in_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x): return self.net(x)

model = LiquidityMax(len(features))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()

print("Optimizing model for high confidence...")
for _ in range(500): 
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward(); optimizer.step()

#6. LIVE FORECAST
model.eval()
with torch.no_grad():
    live_X = scaler.transform(live_row[features].values)
    probs = torch.softmax(model(torch.tensor(live_X, dtype=torch.float32)), dim=1)
    conf, pred = torch.max(probs, dim=1)

mapping = {0: "High Liquidity", 1: "Moderate Liquidity", 2: "Low Liquidity"}

print(f"AS OF DATE:  {live_row.index[-1].date()}")
print(f"PREDICTING:  {mapping[pred.item()]} (for {live_row.index[-1].date() + pd.Timedelta(days=7)})")
print(f"CONFIDENCE:  {conf.item():.2%}")

