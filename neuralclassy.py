# ============================================================
# 1. IMPORTS
# ============================================================
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================================
# 2. DOWNLOAD FUTURES DATA FROM YFINANCE
# ============================================================
ticker = "ZC=F"
df = yf.download(ticker, start="2023-01-01", end="2024-01-01")

df.dropna(inplace=True)
print(df.head())


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
df["return"] = df["Close"].pct_change()
df["volatility"] = df["return"].rolling(10).std()
df["spread_proxy"] = (df["High"] - df["Low"]) / df["Close"]
df["volume"] = df["Volume"]
df["depth_proxy"] = df["Volume"].rolling(5).mean()

df.dropna(inplace=True)

features = ["spread_proxy", "volume", "volatility", "depth_proxy"]
X = df[features].values


# ============================================================
# 4. LABELING LOGIC (3-CLASS LIQUIDITY)
# ============================================================
df["label"] = 0  # normal

df.loc[
    (df["spread_proxy"] < df["spread_proxy"].quantile(0.4)) &
    (df["volume"] > df["volume"].quantile(0.6)),
    "label"
] = 1

df.loc[
    (df["spread_proxy"] > df["spread_proxy"].quantile(0.75)) |
    (df["volatility"] > df["volatility"].quantile(0.75)) |
    (df["depth_proxy"] < df["depth_proxy"].quantile(0.25)),
    "label"
] = 2

y = df["label"].values


# ============================================================
# 5. TRAIN/TEST SPLIT + SCALING
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# ============================================================
# 6. NEURAL NETWORK MODEL
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
# 7. TRAINING LOOP
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
# 8. EVALUATION
# ============================================================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()

print(f"\nTest Accuracy: {accuracy:.2%}")


# ============================================================
# 9. COLOR MAPPING FUNCTION
# ============================================================
def classify_liquidity(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        probs = F.softmax(logits, dim=0)
        pred_class = torch.argmax(probs).item()

        if pred_class == 1:
            return "GREEN (Easy to Sell)"
        elif pred_class == 2:
            return "RED (Hard to Sell)"
        else:
            return "NORMAL"


# ============================================================
# 10. EXAMPLE PREDICTION
# ============================================================
sample = X_test[0]
print("\nSample classification:", classify_liquidity(model, sample))

# ============================================================
# CLEAN SCATTER PRICE CHART WITH DAILY SPANS
# ============================================================

# Convert predictions to numpy
pred_np = predictions.numpy()

# Get test set dates and prices
# Get test set dates and prices (unsorted)
_, X_test_df, _, _ = train_test_split(df, y, test_size=0.2, random_state=42)

# ⭐ SORT BY DATE so the plot follows time correctly
X_test_df = X_test_df.sort_index()

# ⭐ Re-align predictions to the sorted dates
pred_series = pd.Series(pred_np, index=X_test_df.index)

# Extract sorted dates and prices
test_dates = X_test_df.index
test_prices = X_test_df["Close"]

plt.figure(figsize=(16, 8))

# Line plot of price points
plt.plot(test_dates, test_prices, color="black", linewidth=1.5)

# Add vertical spans for each prediction
for date, pred in zip(test_dates, pred_np):
    if pred == 1:
        plt.axvspan(date - pd.Timedelta(hours=36), date + pd.Timedelta(hours=36), color="green", alpha=0.25)
    elif pred == 2:
        plt.axvspan(date - pd.Timedelta(hours=36), date + pd.Timedelta(hours=36), color="red", alpha=0.25)
    # Normal = no highlight

plt.title("Price Chart with Liquidity Stress Highlights (Scatter Only)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot.png', bbox_inches='tight')
plt.show()
