import yfinance as yf
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# NOTESS !
# Every colored bar you see on the graph is the model's attempt to look into the future.
# If you see a Red bar on October 10th, that means the model looked at the data from October 7th 
# and concluded that the market will be hard to sell.
# The graph plots the prediction at the time the event is expected to happen, 
# allowing you to see if price volatility actually occurred during those shaded windows.



ticker = "ZC=F"
start_date = "2023-01-01"
end_date = "2024-01-01"

H = 3  # forecast time period or wahtever (set to 3, 4, or 5)
EPOCHS = 50
LR = 0.001
TRAIN_FRAC = 0.8
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(inplace=True)

print("Raw data head:")
print(df.head())

df["return"] = df["Close"].pct_change()

# Rolling volatility of returns
df["volatility"] = df["return"].rolling(10).std()

# Simple spread proxy from daily bar
df["spread_proxy"] = (df["High"] - df["Low"]) / df["Close"]

# Volume + depth proxy
df["volume"] = df["Volume"]
df["depth_proxy"] = df["Volume"].rolling(5).mean()

# Amihud illiquidity: |return| / dollar_volume
# For futures, dollar volume is approximated by Close*Volume 
df["dollar_volume"] = df["Close"] * df["Volume"]
df["amihud"] = (df["return"].abs() / df["dollar_volume"]).replace([np.inf, -np.inf], np.nan)
df["amihud"] = df["amihud"].rolling(5).mean()

df.dropna(inplace=True)

features = ["spread_proxy", "volume", "volatility", "depth_proxy", "amihud"]



#    liquidity classification 0 = NORMAL, 1 = GREEN (easy), 2 = RED (hard)

df["label"] = 0  # default

# green comes from tighter spread + higher volume
df.loc[
    (df["spread_proxy"] < df["spread_proxy"].quantile(0.40)) &
    (df["volume"] > df["volume"].quantile(0.60)),
    "label"
] = 1

# red comes from wide spread OR high vol OR low depth
df.loc[
    (df["spread_proxy"] > df["spread_proxy"].quantile(0.75)) |
    (df["volatility"] > df["volatility"].quantile(0.75)) |
    (df["depth_proxy"] < df["depth_proxy"].quantile(0.25)),
    "label"
] = 2


#predict some days ahead so it isn't just a glorified classification algorithm like pointed out
df["label_future"] = df["label"].shift(-H)
df.dropna(inplace=True)

X = df[features].values
y = df["label_future"].astype(int).values


split_idx = int(len(df) * TRAIN_FRAC)

X_train_raw = X[:split_idx]
X_test_raw  = X[split_idx:]
y_train = y[:split_idx]
y_test  = y[split_idx:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)


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
    test_logits = model(X_test)
    predictions = torch.argmax(test_logits, dim=1)
    accuracy = (predictions == y_test).float().mean().item()

print(f"\nTest Accuracy (predicting t+{H}): {accuracy:.2%}")


def classify_liquidity_from_scaled(model, x_scaled_1d):
    """
    x_scaled_1d: a 1D numpy array or 1D torch tensor already scaled
    """
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_scaled_1d, dtype=torch.float32)
        logits = model(x_t)
        probs = F.softmax(logits, dim=0)
        pred_class = torch.argmax(probs).item()

    if pred_class == 1:
        return "GREEN (Easy to Sell)"
    elif pred_class == 2:
        return "RED (Hard to Sell)"
    else:
        return "NORMAL"


# Example
sample = X_test[0].numpy()
print("\nSample classification:", classify_liquidity_from_scaled(model, sample))


# 10. PRICE PLOT WITH PREDICTED LIQUIDITY HIGHLIGHTS
#     IMPORTANT: Align predictions to the chronological TEST window

pred_np = predictions.numpy()

test_df = df.iloc[split_idx:].copy()  # corresponds to X_test/y_test
test_dates = test_df.index
test_prices = test_df["Close"]

plt.figure(figsize=(16, 8))
plt.plot(test_dates, test_prices, color="black", linewidth=1.5)

for date, pred in zip(test_dates, pred_np):
    if pred == 1:
        plt.axvspan(date - pd.Timedelta(hours=36), date + pd.Timedelta(hours=36),
                    color="green", alpha=0.25)
    elif pred == 2:
        plt.axvspan(date - pd.Timedelta(hours=36), date + pd.Timedelta(hours=36),
                    color="red", alpha=0.25)

plt.title(f"{ticker} Price with Predicted Liquidity Regimes (Forecast t+{H})")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot.png", bbox_inches="tight")
plt.show()
