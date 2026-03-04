import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# --- CONFIGURATION ---
FND = datetime(2026, 2, 27)
ROLL_DEADLINE = FND - timedelta(days=2) 
ANALYSIS_DATE = datetime(2026, 2, 21) 
TARGET_THRESHOLD = 65.0  

f_ticker, n_ticker = "ZCH26.CBT", "ZCK26.CBT"
main_ticker = "ZC=F"

def get_simulated_data(ticker, cutoff_date):
    # Fix for newer yfinance multi-index behavior
    df = yf.download(ticker, period="2y", multi_level_index=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[df.index <= cutoff_date].dropna()

def get_days_to_expiry_historical(indices):
    expiry_months = [3, 5, 7, 9, 12]
    days_out = []
    for date in indices:
        future_expiries = [datetime(date.year, m, 1) for m in expiry_months if datetime(date.year, m, 1) > date]
        next_expiry = future_expiries[0] if future_expiries else datetime(date.year + 1, 3, 1)
        days_out.append((next_expiry - date).days)
    return np.array(days_out)

def prepare_features(df_in, is_training=False):
    df = df_in.copy()
    df["return"] = df["Close"].pct_change()
    df["vol"] = df["return"].rolling(10).std()
    df["rvol"] = df["Volume"] / df["Volume"].rolling(10).mean()
    if is_training:
        df["days_to_fnd"] = get_days_to_expiry_historical(df.index)
    else:
        df["days_to_fnd"] = (FND - df.index).days
    return df.replace([np.inf, -np.inf], np.nan).dropna()

# Data Download
df_f = get_simulated_data(f_ticker, ANALYSIS_DATE)
df_n = get_simulated_data(n_ticker, ANALYSIS_DATE)
df_main = get_simulated_data(main_ticker, ANALYSIS_DATE)

# Feature Prep
train_df = prepare_features(df_main, is_training=True)
features = ["vol", "rvol", "days_to_fnd"]
q = train_df["vol"].quantile([0.33, 0.66])
train_df["label"] = train_df["vol"].apply(lambda x: 0 if x <= q[0.33] else (2 if x > q[0.66] else 1))
train_df["target"] = train_df["label"].shift(-5)
train_data = train_df.dropna()

# Scaler and Model
scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(train_data[features].values), dtype=torch.float32)
y = torch.tensor(train_data["target"].values, dtype=torch.long)
model = nn.Sequential(nn.Linear(len(features), 32), nn.ReLU(), nn.Linear(32, 3))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Quick Train
for _ in range(100):
    optimizer.zero_grad()
    nn.functional.cross_entropy(model(X), y).backward()
    optimizer.step()

# Inference
model.eval()
live_row = prepare_features(df_f).iloc[-1:]
with torch.no_grad():
    live_X = torch.tensor(scaler.transform(live_row[features].values), dtype=torch.float32)
    state_pred = torch.argmax(model(live_X)).item()


roll_df = pd.DataFrame({'Mar': df_f['Volume'], 'May': df_n['Volume']}).dropna().tail(20)
roll_df['Share'] = (roll_df['May'] / (roll_df['Mar'] + roll_df['May'])) * 100
last_date = roll_df.index[-1]
current_share = roll_df['Share'].iloc[-1]


regime_boost = {0: 0.7, 1: 1.0, 2: 1.5}[state_pred]
days_to_fnd_val = max(1, (FND - last_date).days)
k_final = (0.12 * regime_boost) * (1 + (2.0 / (days_to_fnd_val + 5)))

def predict_logistic(t, start_val, k):
    return 100 / (1 + ((100 - start_val) / start_val) * np.exp(-k * t))


forecast_days = 14
t_range = np.arange(1, forecast_days + 1) 
projections = [current_share] + [predict_logistic(t, current_share, k_final) for t in t_range]
forecast_dates = [last_date] + [last_date + timedelta(days=int(t)) for t in t_range]

final_target = ROLL_DEADLINE
for d, p in zip(forecast_dates, projections):
    if p >= TARGET_THRESHOLD:
        final_target = min(d, ROLL_DEADLINE)
        break


print(f"--- CONNECTED AI ROLL PROJECTION ---")
print(f"Market State: {['STABLE', 'NEUTRAL', 'STRESSED'][state_pred]}")
print(f"Target Date: {final_target.date()}")

fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(roll_df.index, roll_df['Share'], label='Historical May Share %', color='#1f77b4', lw=3)
ax1.plot(forecast_dates, projections, '--', color='#d62728', label='AI Forecast', lw=2)
ax1.axhline(TARGET_THRESHOLD, color='green', linestyle=':', alpha=0.6, label=f'Target {TARGET_THRESHOLD}%')
ax1.scatter(final_target, TARGET_THRESHOLD, color='black', zorder=10, label='Projected Action Date')
ax1.axvline(FND, color='black', linestyle='--', alpha=0.3, label='FND')
ax1.set_ylim(0, 105)
ax1.set_xlim(roll_df.index[0], forecast_dates[-1])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.title(f"Seamless Roll Projection: {['STABLE', 'NEUTRAL', 'STRESSED'][state_pred]} Regime")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()


fig3, ax3 = plt.subplots(figsize=(12, 6))

df_prices = pd.merge(df_f[['Close']], df_n[['Close']], left_index=True, right_index=True, suffixes=('_Mar', '_May'))
df_prices['Spread'] = df_prices['Close_May'] - df_prices['Close_Mar']

ax3.plot(df_prices.index, df_prices['Spread'], color='purple', lw=2, label='May-Mar Spread')
ax3.fill_between(df_prices.index, 0, df_prices['Spread'], where=(df_prices['Spread'] > 0), color='red', alpha=0.2, label='Contango (Cost)')
ax3.fill_between(df_prices.index, 0, df_prices['Spread'], where=(df_prices['Spread'] <= 0), color='green', alpha=0.3, label='Backwardation (Gain)')
ax3.axhline(0, color='black', lw=1.5, ls='--')
ax3.axvline(ANALYSIS_DATE, color='red', linestyle=':', label='Analysis Date')
ax3.set_title(f"Market Term Structure (Roll Cost/Gain Analysis)")
ax3.set_ylabel("Price Spread (Cents)")
ax3.grid(alpha=0.3)
ax3.legend(loc='upper left')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()

plt.show()
last_f = df_prices['Close_Mar'].iloc[-1]
last_n = df_prices['Close_May'].iloc[-1]


months_diff = 2 


# Formula: ((Front - Next) / Front) * (12 / months_diff) * 100
roll_yield_annual = ((last_f - last_n) / last_f) * (12 / months_diff) * 100

recommendation = "STAY (HOLD)"
reasoning = "The roll cost is manageable, and corn may be bottoming."

if roll_yield_annual < -15:
    recommendation = "ROTATE (SELL)"
    reasoning = "High Contango is eroding your capital."
elif state_pred == 2: 
    recommendation = "ROTATE (CAUTION)"
    reasoning = "AI detects high volatility stress. This often leads to wider spreads."


print(f"\n" + "="*30)
print(f"   STRATEGIC RECOMMENDATION")
print(f"="*30)
print(f"Action:      {recommendation}")
print(f"Annual Cost: {roll_yield_annual:.2f}% (Roll Yield)")
print(f"Reason:      {reasoning}")
print("="*30)