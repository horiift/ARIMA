import os
import csv
from datetime import timedelta
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def read_btc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def prepare_series(df: pd.DataFrame):
    s = df.set_index("Date")["Open"].copy()
    s = pd.to_numeric(s, errors="coerce").dropna()
    s_pos = s[s > 0].copy()
    s_pos.index = s_pos.index.normalize()
    s_log = np.log(s_pos)
    return s_pos, s_log

def rw_drift_manual(train_log: pd.Series, steps: int) -> np.ndarray | None:
    if len(train_log) < 2:
        return None
    diffs = train_log.diff().dropna()
    if diffs.empty:
        return None
    mu = float(diffs.mean())
    last = float(train_log.iloc[-1])
    horizons = np.arange(1, steps + 1, dtype=float)
    fc_log = last + mu * horizons
    return np.exp(fc_log)

def arima_drift_forecast_last_window(series_log: pd.Series, window_days: int, steps: int) -> np.ndarray | None:
    if len(series_log) < window_days:
        return None
    train = series_log.iloc[-window_days:]
    try:
        model = ARIMA(
            train,
            order=(0, 1, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit()
        fc_log = fitted.forecast(steps=steps)
        return np.exp(np.asarray(fc_log))
    except Exception:
        return rw_drift_manual(train, steps)

def main():
    input_csv = "btc_daily.csv"
    output_dir = "1-ARIMA"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "predictions.csv")

    df = read_btc_csv(input_csv)

    # Validação inicia em 2017-07-01 até o fim disponível
    desired_start = pd.Timestamp("2017-07-01")

    s_pos, s_log = prepare_series(df)

    data_start = s_pos.index.min()
    data_end = s_pos.index.max()
    start_date = max(desired_start, data_start)
    end_date = data_end

    horizons = list(range(1, 8))
    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
        print("[INFO] Não há dados no intervalo solicitado.")
        fieldnames = (
            ["date", "dow", "window_days", "open_t"]
            + sum(([f"forecast_t+{h}", f"real_t+{h}",
                    f"signal_retrain_t+{h}", f"return_retrain_t+{h}",
                    f"signal_daily_t+{h}", f"return_daily_t+{h}"] for h in horizons), [])
            + sum(([f"signal_vs_forecast7_open_t+{k}", f"return_vs_forecast7_open_t+{k}"] for k in horizons), [])
        )
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    # Janelas de treinamento solicitadas (ajuste: remover 30 e incluir 28)
    windows = [7, 14, 28, 60, 120, 360, 720]
    max_horizon = 7

    rows = []
    eval_dates = s_pos.index[(s_pos.index >= start_date) & (s_pos.index <= end_date)]

    for current_day in eval_dates:
        dow = int(current_day.weekday())
        train_series_log = s_log.loc[:current_day]
        if len(train_series_log) < 2:
            continue

        open_t = float(s_pos.loc[current_day])
        if not np.isfinite(open_t) or open_t <= 0:
            continue

        for w in windows:
            fc = arima_drift_forecast_last_window(train_series_log, w, max_horizon)
            if fc is None:
                continue

            row = {
                "date": current_day.strftime("%Y-%m-%d"),
                "dow": dow,
                "window_days": w,
                "open_t": open_t,
            }

            # Colunas originais (não mexer na lógica)
            for h in horizons:
                y_hat = float(fc[h - 1]) if np.isfinite(fc[h - 1]) else np.nan
                row[f"forecast_t+{h}"] = y_hat

                future_day = current_day + timedelta(days=h)
                real_future = float(s_pos.loc[future_day]) if future_day in s_pos.index else np.nan
                row[f"real_t+{h}"] = real_future

                # Inicializa outputs para evitar KeyError
                row[f"signal_retrain_t+{h}"] = ""
                row[f"return_retrain_t+{h}"] = ""
                row[f"signal_daily_t+{h}"] = ""
                row[f"return_daily_t+{h}"] = ""

                # Sinal/retorno retrain
                if np.isfinite(y_hat) and open_t > 0:
                    pred_ret_retrain = (y_hat / open_t) - 1.0
                    signal_retrain = 1 if pred_ret_retrain >= 0 else -1
                    row[f"signal_retrain_t+{h}"] = int(signal_retrain)

                    if np.isfinite(real_future):
                        actual_ret_retrain = (real_future / open_t) - 1.0
                        row[f"return_retrain_t+{h}"] = float(signal_retrain * actual_ret_retrain)

                # Sinal/retorno daily
                prev_day = current_day + timedelta(days=h - 1)
                prev_open = float(s_pos.loc[prev_day]) if prev_day in s_pos.index else np.nan
                if np.isfinite(y_hat) and np.isfinite(prev_open) and prev_open > 0:
                    signal_daily = 1 if y_hat >= prev_open else -1
                    row[f"signal_daily_t+{h}"] = int(signal_daily)
                    if np.isfinite(real_future):
                        ret_daily = (real_future - prev_open) / prev_open
                        row[f"return_daily_t+{h}"] = float(signal_daily * ret_daily)

            # Novas colunas: comparação usando forecast_t+7 contra open_{t+k-1}, k=1..7
            forecast_t7 = row.get("forecast_t+7", np.nan)
            for k in horizons:
                # Inicializa para evitar KeyError
                row[f"signal_vs_forecast7_open_t+{k}"] = ""
                row[f"return_vs_forecast7_open_t+{k}"] = ""

                prev_day_k = current_day + timedelta(days=k - 1)
                day_k = current_day + timedelta(days=k)
                prev_open_k = float(s_pos.loc[prev_day_k]) if prev_day_k in s_pos.index else np.nan
                open_k = float(s_pos.loc[day_k]) if day_k in s_pos.index else np.nan

                if np.isfinite(forecast_t7) and np.isfinite(prev_open_k) and prev_open_k > 0:
                    sig_vs_f7 = 1 if forecast_t7 >= prev_open_k else -1
                    row[f"signal_vs_forecast7_open_t+{k}"] = int(sig_vs_f7)

                    if np.isfinite(open_k):
                        ret_vs_f7 = (open_k - prev_open_k) / prev_open_k
                        row[f"return_vs_forecast7_open_t+{k}"] = float(sig_vs_f7 * ret_vs_f7)

            rows.append(row)

    rows.sort(key=lambda r: (r["date"], r["window_days"]))

    fieldnames = (
        ["date", "dow", "window_days", "open_t"]
        + sum(([f"forecast_t+{h}", f"real_t+{h}",
                f"signal_retrain_t+{h}", f"return_retrain_t+{h}",
                f"signal_daily_t+{h}", f"return_daily_t+{h}"] for h in horizons), [])
        + sum(([f"signal_vs_forecast7_open_t+{k}", f"return_vs_forecast7_open_t+{k}"] for k in horizons), [])
    )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Linhas geradas: {len(rows)} no arquivo {output_csv}")
    print(f"[INFO] Intervalo efetivo de avaliação: {start_date.date()} a {end_date.date()}")

if __name__ == "__main__":
    main()