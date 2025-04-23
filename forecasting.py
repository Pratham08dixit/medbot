import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sqlite3

# ---------------------------
# 1. Load Usage Data
# ---------------------------
def load_usage_data(csv_file):
    df = pd.read_csv(csv_file)
    df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
    return df.sort_values('Day')

# ---------------------------
# 1b. Load Inventory Data
# ---------------------------
def load_inventory(csv_file):
    inv_df = pd.read_csv(csv_file)
    cols = inv_df.columns.tolist()
    if len(cols) < 2:
        raise ValueError("Inventory CSV must have at least two columns: drug and inventory")
    key_col, val_col = cols[0], cols[1]
    inv_map = {
        str(k).strip().lower(): float(v)
        for k, v in zip(inv_df[key_col], inv_df[val_col])
        if pd.notnull(k)
    }
    return inv_map

# ---------------------------
# 2. Feature Engineering: Create Lag Features
# ---------------------------
def create_lag_features(df, target_col, lags=10):
    df_fe = df.copy()
    for lag in range(1, lags + 1):
        df_fe[f"{target_col}_lag_{lag}"] = df_fe[target_col].shift(lag)
    return df_fe.dropna().reset_index(drop=True)

# ---------------------------
# 3. Forecast Function
# ---------------------------
def forecast_for_drug(df, drug, inv_map, lags=10, horizon=3, stock_target=40):
    """
    Train an XGBoost model on the last `lags` days of usage to forecast the next `horizon` days.
    Returns day1, day2, day3 forecasts (rounded ints), inventory, reorder, MAE/RMSE.
    """
    df_fe = create_lag_features(df, target_col=drug, lags=lags)
    train = df_fe.iloc[:-1]
    last_row = df_fe.iloc[-1]
    feat_cols = [f"{drug}_lag_{i}" for i in range(1, lags+1)]
    X_train = train[feat_cols]
    y_train = train[drug]

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # In-sample error metrics
    y_pred_train = model.predict(X_train)
    mae = float(mean_absolute_error(y_train, y_pred_train))
    rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

    # Multi-step forecast
    last_feats = last_row[feat_cols].values
    next_days = []
    for _ in range(horizon):
        pred = float(model.predict(last_feats.reshape(1, -1))[0])
        pred_int = int(round(pred))
        next_days.append(pred_int)
        # roll forward
        last_feats = np.roll(last_feats, 1)
        last_feats[0] = pred_int

    # Compute inventory logic
    current_inv = inv_map.get(drug.lower(), 0.0)
    total_forecast = float(sum(next_days))
    reorder_qty = float(max(total_forecast + stock_target - current_inv, 0.0))

    return {
        'drug': drug,
        'day1': next_days[0],
        'day2': next_days[1],
        'day3': next_days[2],
        'total': total_forecast,
        'inventory': current_inv,
        'reorder': reorder_qty,
        'mae': mae,
        'rmse': rmse
    }
