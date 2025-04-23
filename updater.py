# updater.py

import pandas as pd
import sqlite3
import numpy as np
from forecasting import load_usage_data, load_inventory, forecast_for_drug

USAGE_CSV     = "sample_drug_usage.csv"
INVENTORY_CSV = "current_inventory.csv"
DB_FILE       = "forecasts.db"

def update_history_and_db():
    # 1) Load history & inventory mapping
    usage_df = load_usage_data(USAGE_CSV)
    inv_map   = load_inventory(INVENTORY_CSV)
    inv_df = pd.read_csv(INVENTORY_CSV)

    # 2) Connect to DB and ensure table exists with 3-day forecast columns
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS forecasts (
        drug TEXT PRIMARY KEY,
        forecast_day1 REAL,
        forecast_day2 REAL,
        forecast_day3 REAL,
        total_forecast REAL,
        current_inventory REAL,
        reorder_quantity REAL
      );
    ''')
    conn.commit()

    # 3) Forecast for each drug
    drugs    = [c for c in usage_df.columns if c.lower() != "day"]
    last_day = int(usage_df["Day"].max())
    predictions = {}

    maes, rmses = [], []

    for drug in drugs:
        res = forecast_for_drug(usage_df, drug, inv_map, horizon=3)

        total_forecast = int(round(res["day1"] + res["day2"] + res["day3"]))
        predictions[drug] = {
            'day1': res["day1"],
            'day2': res["day2"],
            'day3': res["day3"],
            'total': total_forecast,
            'initial_inv': int(round(inv_map.get(drug.lower(), 0))),
            'reorder': int(round(res["reorder"]))
        }

        # collect metrics for overall
        maes.append(res['mae'])
        rmses.append(res['rmse'])

        # upsert into forecasts table
        cursor.execute("""
            REPLACE INTO forecasts
            (drug, forecast_day1, forecast_day2, forecast_day3,
             total_forecast, current_inventory, reorder_quantity)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            str(res["drug"]),
            int(res["day1"]),
            int(res["day2"]),
            int(res["day3"]),
            total_forecast,
            int(round(res["inventory"])),
            int(round(res["reorder"]))
        ))

    conn.commit()
    conn.close()

    # 4) Append 3 forecasted days to usage CSV
    new_rows = []
    for offset in (1, 2, 3):
        row = {"Day": last_day + offset}
        for drug in drugs:
            row[drug] = int(predictions[drug][f'day{offset}'])
        new_rows.append(row)

    updated_df = pd.concat([usage_df, pd.DataFrame(new_rows)], ignore_index=True)
    for drug in drugs:
        updated_df[drug] = updated_df[drug].round(0).astype(int)
    updated_df.to_csv(USAGE_CSV, index=False)

    # 5) Update current inventory in inventory CSV
    key_col, val_col = inv_df.columns[0], inv_df.columns[1]
    for idx, row in inv_df.iterrows():
        drug_name = str(row[key_col]).strip()
        info = predictions.get(drug_name)
        if info:
            new_qty = abs(info['total'] - (info['initial_inv'] + info['reorder']))
            inv_df.at[idx, val_col] = int(round(new_qty))
    inv_df.to_csv(INVENTORY_CSV, index=False)

    # 6) Compute & print overall MAE and RMSE across all drugs
    overall_mae  = float(np.mean(maes))  if maes  else float('nan')
    overall_rmse = float(np.mean(rmses)) if rmses else float('nan')
    print(f"\nOverall MAE across all drugs:  {overall_mae:.2f}")
    print(f"Overall RMSE across all drugs: {overall_rmse:.2f}")

    print(f"\nAppended days {last_day+1}, {last_day+2} & {last_day+3} to {USAGE_CSV}, "
          f"updated DB, and recalculated {INVENTORY_CSV}.")

if __name__ == "__main__":
    update_history_and_db()
