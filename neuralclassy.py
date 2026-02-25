
# ============================================================
# 1. IMPORTS
# ============================================================
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ============================================================
# 2. DOWNLOAD FUTURES DATA
# ============================================================
ticker = "ZC=F"
df = yf.download(ticker, start="2020-01-01", end="2024-01-01")
df.dropna(inplace=True)

# FIX 1: Flatten MultiIndex columns
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# FIX 2: Ensure Volume is a 1D Series
if isinstance(df["Volume"], pd.DataFrame):
    df["Volume"] = df["Volume"].iloc[:, 0]
df["Volume"] = df["Volume"].astype(float)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
df["return"] = df["Close"].pct_change()
df["volatility"] = df["return"].rolling(10).std()
df["spread_proxy"] = (df["High"] - df["Low"]) / df["Close"]
df["depth_proxy"] = df["Volume"].rolling(5).mean()

# Raw Amihud
df["amihud_raw"] = (df["return"].abs()) / (df["Close"] * df["Volume"])

# SMOOTHED AMIHUD (5-day rolling mean)
df["amihud"] = df["amihud_raw"].rolling(5).mean()


# ============================================================
# 4. CLEAN BEFORE LABELING
# ============================================================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()


# ============================================================
# 5. CREATE 3 LIQUIDITY REGIMES USING SMOOTHED AMIHUD
# ============================================================
df["label"] = pd.qcut(df["amihud"], q=3, labels=[0, 1, 2]).astype(int)


# ============================================================
# 6. SHIFT LABELS FOR 7-DAY FORECASTING
# ============================================================
df["label_next_week"] = df["label"].shift(-7)


# ============================================================
# 7. FINAL CLEANING (AFTER SHIFTING)
# ============================================================
features = ["spread_proxy", "Volume", "volatility", "depth_proxy", "amihud"]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + ["label_next_week"])


# ============================================================
# 8. PREPARE FEATURES + TARGET
# ============================================================
X = df[features].values
y = df["label_next_week"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)


# ============================================================
# 9. TRAIN/TEST SPLIT (CHRONOLOGICAL)
# ============================================================
split_idx = int(len(df) * 0.8)

X_train = X_tensor[:split_idx]
X_test = X_tensor[split_idx:]
y_train = y_tensor[:split_idx]
y_test = y_tensor[split_idx:]

dates_test = df.index[split_idx:]
prices_test = df["Close"].iloc[split_idx:]


# ============================================================
# 10. MODEL
# ============================================================
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


# ============================================================
# 11. TRAINING LOOP
# ============================================================
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


# ============================================================
# 12. EVALUATION
# ============================================================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()

print(f"\n7-Day Ahead Forecast Accuracy: {accuracy:.2%}")


# ============================================================
# 13. PLOT RESULTS (THICK GREEN/RED REGIME BARS)
# ============================================================
pred_np = predictions.numpy()

plt.figure(figsize=(16, 8))
plt.plot(dates_test, prices_test, color="black", linewidth=3)

# THICKER REGIME BARS (1-day width)
for date, pred in zip(dates_test, pred_np):
    if pred == 0:
        plt.axvspan(date, date + pd.Timedelta(days=1), color="green", alpha=0.25)
    elif pred == 2:
        plt.axvspan(date, date + pd.Timedelta(days=1), color="red", alpha=0.25)

plt.title("7-Day Ahead Liquidity Forecast (Smoothed Amihud Regimes)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Volatility Plot
plt.figure(figsize=(16, 6))
plt.plot(df.index, df["volatility"], color="purple", linewidth=3)
plt.title(f"{ticker} Rolling Volatility (10-Day Window)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("volatility_plot.png", bbox_inches="tight")
plt.show()


# Center day of the test-set plot
center_date = dates_test[int(len(dates_test) / 2)]
print("Center of plot:", center_date)
center_values = df.loc[center_date, ["Close", "High", "Volume", "label_next_week"]]

print("\nValues for center day:")
print("Close:", center_values["Close"])
print("High:", center_values["High"])
print("Volume:", center_values["Volume"])
print("Regime:", int(center_values["label_next_week"]))
