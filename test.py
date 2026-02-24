import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


ticker = "ZC=F"
start_date = "2023-01-01"
end_date = "2026-02-24"  
H = 3 
EPOCHS = 50
LR = 0.001
TRAIN_FRAC = 0.8
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(inplace=True)


df["return"] = df["Close"].pct_change()


df["vol_pct"] = df["return"].rolling(10).std() * np.sqrt(252) * 100
df["vol_trend_fast"] = df["vol_pct"].rolling(5).mean()
df["vol_trend_slow"] = df["vol_pct"].rolling(20).mean()
df["vol_momentum"] = df["vol_trend_fast"] - df["vol_trend_slow"]


df["spread_proxy"] = (df["High"] - df["Low"]) / df["Close"]
df["volume"] = df["Volume"]
df["depth_proxy"] = df["Volume"].rolling(5).mean()
df["dollar_volume"] = df["Close"] * df["Volume"]
df["amihud"] = (df["return"].abs() / df["dollar_volume"]).replace([np.inf, -np.inf], np.nan)
df["amihud"] = df["amihud"].rolling(5).mean()

df.dropna(inplace=True)


features = ["spread_proxy", "volume", "vol_pct", "vol_momentum", "depth_proxy", "amihud"]


df["label"] = 0

df.loc[(df["spread_proxy"] < df["spread_proxy"].quantile(0.40)) & (df["volume"] > df["volume"].quantile(0.60)), "label"] = 1

df.loc[(df["spread_proxy"] > df["spread_proxy"].quantile(0.75)) | (df["vol_pct"] > df["vol_pct"].quantile(0.75)) | (df["depth_proxy"] < df["depth_proxy"].quantile(0.25)), "label"] = 2

df["label_future"] = df["label"].shift(-H)
df.dropna(inplace=True)

X = df[features].values
y = df["label_future"].astype(int).values

split_idx = int(len(df) * TRAIN_FRAC)
X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train_raw), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


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
    def forward(self, x): return self.layers(x)

model = LiquidityClassifier(input_dim=len(features))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    predictions = torch.argmax(model(X_test), dim=1)
accuracy = (predictions == y_test).float().mean().item()
print(f"\nTest Accuracy (predicting t+{H}): {accuracy:.2%}")


plt.figure(figsize=(16, 6))
plt.plot(df.index, df["vol_trend_fast"], label="Short-term Vol (5d)", color="orange")
plt.plot(df.index, df["vol_trend_slow"], label="Long-term Vol (20d)", color="blue", linestyle="--")
plt.title(f"{ticker} Annualized Volatility Trends (2026)")
plt.ylabel("Volatility (%)")
plt.legend()
plt.show()