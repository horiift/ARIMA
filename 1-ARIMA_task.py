import os
import sys
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from dateutil.relativedelta import relativedelta

# ------------------------------------------------------------
# Ajuste solicitado:
# - Nos eixos X das tabelas / heatmaps, onde for coluna semanal,
#   mostrar em DUAS LINHAS:
#        Semanal
#        (Seg +7)   etc.
#   Isto indica o horizonte de +7 dias.
# - Colunas semanais: "Semanal\n(Seg +7)", ..., "Semanal\n(Dom +7)"
# - Coluna do retraino diário permanece: "Retreino Diário (+1)"
# - Demais traduções e lógica anteriores mantidas.
#
# Alteração adicional (pedido atual):
# - Gerar somente o heatmap do período de 8 anos e os CSVs relacionados
#   (período fixo 2018-01-01 a 2025-12-31).
# ------------------------------------------------------------

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DISPLAY_WEEKDAY_MAP = {
    "Mon": "Seg",
    "Tue": "Ter",
    "Wed": "Qua",
    "Thu": "Qui",
    "Fri": "Sex",
    "Sat": "Sáb",
    "Sun": "Dom",
}
# Pre-compute inverse map once instead of rebuilding in each function call
INVERSE_DISPLAY_WEEKDAY_MAP = {v: k for k, v in DISPLAY_WEEKDAY_MAP.items()}

def semanal_label(wd_internal: str) -> str:
    """Retorna o rótulo multi-linha para a coluna semanal de um dia interno."""
    return f"Semanal\n({DISPLAY_WEEKDAY_MAP[wd_internal]} +7)"

def is_semanal_label(col: str) -> bool:
    """Verifica se a coluna segue o padrão multi-linha Semanal."""
    return col.startswith("Semanal\n(") and col.endswith("+7)")

def extract_wd_from_label(col: str) -> str | None:
    """Extrai nome interno (Mon..Sun) a partir do rótulo multi-linha."""
    if not is_semanal_label(col):
        return None
    # Padrão: Semanal\n(<Abrev +7)
    # Exemplo: Semanal\n(Seg +7)
    try:
        inside = col.split("\n", 1)[1]  # (Seg +7)
        inside = inside.strip("()")     # Seg +7
        abbrev = inside.split("+7")[0].strip()  # Seg
        # Use pre-computed inverse map for better performance
        return INVERSE_DISPLAY_WEEKDAY_MAP.get(abbrev)
    except Exception:
        return None

def build_diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "vermelho_branco_azul",
        [(1, 0, 0), (1, 1, 1), (0, 0, 1)],
        N=256
    )

def detect_horizons(columns, prefix):
    escaped = re.escape(prefix)
    pattern = re.compile(rf"^{escaped}t\+(\d+)$")
    return sorted([int(m.group(1)) for c in columns if (m := pattern.match(c))])

def product_return(returns: pd.Series) -> float:
    # Filter out NaN values using vectorized operation
    valid_returns = returns.dropna()
    if valid_returns.empty:
        return 0.0
    # Check for catastrophic loss using vectorized comparison
    if (valid_returns <= -0.999999).any():
        # Capital goes to 0 after catastrophic loss (original behavior)
        return -100.0
    # Vectorized computation of cumulative product
    capital = np.prod(1 + valid_returns.values)
    return (capital - 1.0) * 100.0

def make_norm(values):
    if values.size == 0:
        return TwoSlopeNorm(vmin=-1e-6, vcenter=0.0, vmax=1e-6)
    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    if not np.isfinite(min_val): min_val = 0.0
    if not np.isfinite(max_val): max_val = 0.0
    span = max(abs(min_val), abs(max_val))
    if span == 0: span = 1e-6
    return TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

def compute_comp_metric(df_terms: pd.DataFrame, label: str):
    records = []
    for w, group in df_terms.groupby("window_days"):
        comp_ret = product_return(group.sort_values(["target_date", "horizon_h"])["return_value"])
        records.append({
            "window_days": w,
            label: comp_ret,
            f"{label} N_terms": len(group)
        })
    return pd.DataFrame(records).set_index("window_days")

def build_terms_weekly_for_day(group_df, retrain_horizons, cutoff_target):
    h_full = 7
    if h_full not in retrain_horizons:
        return pd.DataFrame()
    if group_df.empty:
        return pd.DataFrame()
    
    # Vectorized approach: compute all target dates at once
    return_col = f"return_retrain_t+{h_full}"
    if return_col not in group_df.columns:
        return pd.DataFrame()
    
    # Compute target dates without modifying original dataframe
    target_full = group_df["date"] + pd.Timedelta(days=h_full)
    cutoff_limit = cutoff_target - pd.Timedelta(days=1)
    
    # Filter rows using vectorized operations
    mask = (target_full <= cutoff_limit) & group_df[return_col].notna()
    filtered = group_df.loc[mask]
    target_full_filtered = target_full.loc[mask]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Build result DataFrame using vectorized operations
    result = pd.DataFrame({
        "window_days": filtered["window_days"].astype(int),
        "base_date": filtered["date"].dt.date.astype(str),
        "horizon_h": h_full,
        "target_date": target_full_filtered.dt.date.astype(str),
        "return_value": filtered[return_col].astype(float),
        "factor": 1.0 + filtered[return_col].astype(float),
        "source": "validacao_semanal"
    })
    
    return result

def build_terms_ret_diario(dsel, stop_date):
    if stop_date is None or "return_daily_t+1" not in dsel.columns:
        return pd.DataFrame()
    validation_start = dsel["date"].min()
    max_base_date = pd.Timestamp(stop_date) - pd.Timedelta(days=1)
    subset = dsel[(dsel["date"] >= validation_start) & (dsel["date"] <= max_base_date)].sort_values(["window_days", "date"])
    if subset.empty:
        return pd.DataFrame()
    
    # Compute target dates without modifying original dataframe
    target_date_ts = subset["date"] + pd.Timedelta(days=1)
    
    # Filter rows using vectorized operations
    mask = (target_date_ts.dt.date <= stop_date) & subset["return_daily_t+1"].notna()
    filtered = subset.loc[mask]
    target_date_filtered = target_date_ts.loc[mask]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Build result DataFrame using vectorized operations
    result = pd.DataFrame({
        "window_days": filtered["window_days"].astype(int),
        "base_date": filtered["date"].dt.date.astype(str),
        "horizon_h": 1,
        "target_date": target_date_filtered.dt.date.astype(str),
        "return_value": filtered["return_daily_t+1"].astype(float),
        "factor": 1.0 + filtered["return_daily_t+1"].astype(float),
        "source": "ret_diario"
    })
    
    return result

def compute_btc_appreciation(dsel: pd.DataFrame):
    if "open_t" not in dsel.columns:
        return None
    op_series = pd.to_numeric(dsel["open_t"], errors="coerce")
    tmp = pd.DataFrame({"date": dsel["date"], "open_t": op_series}).dropna(subset=["open_t"])
    if tmp.empty:
        return None
    px = tmp.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")["open_t"].sort_index()
    if px.empty:
        return None
    start_dt = px.index.min()
    end_dt = px.index.max()
    p0, p1 = float(px.loc[start_dt]), float(px.loc[end_dt])
    if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
        return None
    return {
        "start_date": start_dt.date().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "start_open": p0,
        "end_open": p1,
        "appreciation_pct": (p1 / p0 - 1.0) * 100.0
    }

def compute_appreciation_between_dates(dsel: pd.DataFrame, date_open: pd.Timestamp | None, date_close: pd.Timestamp | None) -> float | None:
    if date_open is None or date_close is None:
        return None
    if "open_t" not in dsel.columns:
        return None
    open_map = dsel.drop_duplicates(subset=["date"]).set_index("date")["open_t"]
    if date_open not in open_map.index or date_close not in open_map.index:
        return None
    p0 = open_map.loc[date_open]; p1 = open_map.loc[date_close]
    if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
        return None
    return (p1 / p0 - 1.0) * 100.0

def render_period(df, daily_horizons, retrain_horizons, period_start, period_end, label, out_dir):
    last_recorded_date = pd.to_datetime(df["date"]).max()
    if pd.isna(last_recorded_date):
        print(f"[ERROR] Sem dados.")
        return
    validation_start = period_start
    validation_end = min(period_end, last_recorded_date)
    if validation_end < validation_start:
        print(f"[WARN] {label}: fim < início.")
        return
    cutoff_target = validation_end + pd.Timedelta(days=1)

    start_weekday_name = WEEKDAY_NAMES[int(validation_start.weekday())]

    dsel = df[(df["date"] >= validation_start) & (df["date"] <= validation_end)].copy()
    if dsel.empty:
        print(f"[WARN] {label}: sem linhas de validação.")
        return

    dsel["date"] = pd.to_datetime(dsel["date"], errors="coerce")
    dsel["dow"] = pd.to_numeric(dsel["dow"], errors="coerce")
    dsel["window_days"] = pd.to_numeric(dsel["window_days"], errors="coerce")
    dsel = dsel.dropna(subset=["date", "dow", "window_days"])

    for h in daily_horizons:
        dsel[f"return_daily_t+{h}"] = pd.to_numeric(dsel.get(f"return_daily_t+{h}"), errors="coerce")
    for h in retrain_horizons:
        dsel[f"return_retrain_t+{h}"] = pd.to_numeric(dsel.get(f"return_retrain_t+{h}"), errors="coerce")
    if "open_t" in dsel.columns:
        dsel["open_t"] = pd.to_numeric(dsel["open_t"], errors="coerce")

    csv_dir = os.path.join(out_dir, "CSV", label)
    os.makedirs(csv_dir, exist_ok=True)

    btc_summary = compute_btc_appreciation(dsel)
    btc_text = f"BTC {btc_summary['start_date']}→{btc_summary['end_date']}: {btc_summary['appreciation_pct']:+.2f}%" if btc_summary else "Valorização do BTC: N/A"

    uniform_base_cutoff = validation_end - pd.Timedelta(days=7)

    candidate_base_dates = {}
    for idx, wd in enumerate(WEEKDAY_NAMES):
        day_all = dsel[dsel["dow"] == idx]
        if day_all.empty:
            candidate_base_dates[wd] = []
            continue
        day_f = day_all[day_all["date"] < uniform_base_cutoff].sort_values("date")
        if day_f.empty:
            candidate_base_dates[wd] = []
            continue
        full_rows = day_f[(day_f["date"] + pd.Timedelta(days=7)) <= validation_end]
        candidate_base_dates[wd] = list(full_rows["date"].sort_values())

    counts = {wd: len(lst) for wd, lst in candidate_base_dates.items()}
    counts_except_mon_tue = [counts[w] for w in WEEKDAY_NAMES if w not in ["Mon", "Tue"]]
    target_count = min(counts_except_mon_tue) if counts_except_mon_tue else 0
    print(f"[INFO] {label}: contagens={counts} alvo_uniforme={target_count}")

    first_weekly_base_dates = {}
    last_weekly_target_dates = {}
    weekly_comp_series = {}

    if target_count > 0:
        for idx, wd in enumerate(WEEKDAY_NAMES):
            base_list = candidate_base_dates[wd]
            if not base_list:
                first_weekly_base_dates[wd] = None
                last_weekly_target_dates[wd] = None
                continue
            sel = base_list[:target_count]
            subset = dsel[(dsel["dow"] == idx) & (dsel["date"].isin(sel))].sort_values(["window_days", "date"])
            weekly_terms = build_terms_weekly_for_day(subset, retrain_horizons, cutoff_target)
            weekly_terms.to_csv(os.path.join(csv_dir, f"{wd}_semanal_termos_uniforme.csv"), index=False)
            if sel:
                first_weekly_base_dates[wd] = sel[0].date()
                last_weekly_target_dates[wd] = (sel[-1] + pd.Timedelta(days=7)).date()
            else:
                first_weekly_base_dates[wd] = None
                last_weekly_target_dates[wd] = None
            comp_weekly = compute_comp_metric(weekly_terms, "Semanal (%)")
            if not comp_weekly.empty:
                weekly_comp_series[wd] = comp_weekly["Semanal (%)"].rename(wd)
    else:
        for wd in WEEKDAY_NAMES:
            first_weekly_base_dates[wd] = None
            last_weekly_target_dates[wd] = None

    semanal_heat = pd.concat(weekly_comp_series.values(), axis=1).sort_index(ascending=False) if weekly_comp_series else pd.DataFrame()

    stop_date_start_wd = last_weekly_target_dates.get(start_weekday_name)
    ret_diario_terms = build_terms_ret_diario(dsel, stop_date_start_wd)
    ret_diario_terms.to_csv(os.path.join(csv_dir, f"{start_weekday_name}_retreino_diario_termos.csv"), index=False)
    if ret_diario_terms.empty:
        ret_diario_heat = pd.DataFrame()
        ret_diario_pct_numeric = np.nan
    else:
        comp_ret_diario = compute_comp_metric(ret_diario_terms, "Retreino Diário (+1) (%)").reset_index()
        ret_diario_heat = comp_ret_diario.set_index("window_days")[["Retreino Diário (+1) (%)"]].sort_index(ascending=False)
        comp_ret_diario.to_csv(os.path.join(csv_dir, f"{start_weekday_name}_retreino_diario_resumo.csv"), index=False)

    btc_semanal_numeric = {}
    btc_semanal_str = {}
    for wd in WEEKDAY_NAMES:
        o = first_weekly_base_dates.get(wd)
        c = last_weekly_target_dates.get(wd)
        if o is not None and c is not None:
            pct = compute_appreciation_between_dates(dsel, pd.Timestamp(o), pd.Timestamp(c))
            btc_semanal_numeric[wd] = pct if pct is not None else np.nan
            btc_semanal_str[wd] = f"{pct:+.2f}%" if pct is not None else ""
        else:
            btc_semanal_numeric[wd] = np.nan
            btc_semanal_str[wd] = ""

    if stop_date_start_wd is not None:
        btc_ret_diario_pct = compute_appreciation_between_dates(dsel, validation_start, pd.Timestamp(stop_date_start_wd))
        ret_diario_pct_numeric = btc_ret_diario_pct if btc_ret_diario_pct is not None else np.nan
        ret_diario_btc_str = f"{btc_ret_diario_pct:+.2f}%" if btc_ret_diario_pct is not None else ""
    else:
        ret_diario_pct_numeric = np.nan
        ret_diario_btc_str = ""

    combined_cols = []
    if not semanal_heat.empty:
        combined_cols.append(semanal_heat)
    if not ret_diario_heat.empty:
        combined_cols.append(ret_diario_heat)
    combined_heat = pd.concat(combined_cols, axis=1) if combined_cols else pd.DataFrame()

    if not combined_heat.empty:
        new_cols = []
        for col in combined_heat.columns:
            if col in WEEKDAY_NAMES:
                new_cols.append(semanal_label(col))
            elif "Retreino Diário (+1)" in col:
                new_cols.append("Retreino Diário (+1)")
            else:
                new_cols.append(col)
        combined_heat.columns = new_cols

    n_windows = dsel["window_days"].nunique()
    cell_height_factor = 0.30
    base_fig_height = max(3.5, n_windows * cell_height_factor)
    height_ratios = [0.70, 1.0, 1.0]
    fig_width = 12.0
    fig_height_total = base_fig_height * sum(height_ratios)

    sns.set_theme(style="white", font_scale=0.9)
    cmap = build_diverging_cmap()

    fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height_total),
                             gridspec_kw={"height_ratios": height_ratios})
    try:
        fig.suptitle(f"{label} — {btc_text}", fontsize=13.0, y=1.01)
    except Exception:
        fig.text(0.5, 1.005, f"{label} — {btc_text}", ha="center", va="top", fontsize=13.0)

    # Tabela superior
    table_cols = [semanal_label(w) for w in WEEKDAY_NAMES] + ["Retreino Diário (+1)"]
    ax_table = axes[0]
    table_df = pd.DataFrame(index=["Início", "Fim", "Valorização BTC"], columns=table_cols, dtype=object)
    for wd in WEEKDAY_NAMES:
        col_name = semanal_label(wd)
        table_df.loc["Início", col_name] = first_weekly_base_dates.get(wd).isoformat() if first_weekly_base_dates.get(wd) else ""
        table_df.loc["Fim", col_name] = last_weekly_target_dates.get(wd).isoformat() if last_weekly_target_dates.get(wd) else ""
        table_df.loc["Valorização BTC", col_name] = btc_semanal_str.get(wd, "")
    table_df.loc["Início", "Retreino Diário (+1)"] = validation_start.date().isoformat()
    table_df.loc["Fim", "Retreino Diário (+1)"] = stop_date_start_wd.isoformat() if stop_date_start_wd else ""
    table_df.loc["Valorização BTC", "Retreino Diário (+1)"] = ret_diario_btc_str

    dummy_table = pd.DataFrame(np.zeros_like(table_df.values, dtype=float),
                               index=table_df.index, columns=table_df.columns)
    sns.heatmap(dummy_table, cmap="Greys", vmin=0, vmax=0,
                annot=table_df, fmt="", linewidths=0.5, linecolor="gray",
                cbar=False, ax=ax_table)
    ax_table.set_title("Início / Fim / Valorização do bitcoin no intervalo", fontsize=10.5, pad=6)
    ax_table.set_xlabel("")
    ax_table.set_ylabel("")
    plt.setp(ax_table.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax_table.get_yticklabels(), rotation=0, ha="right")

    # Heatmap principal
    ax_main = axes[1]
    if combined_heat.empty:
        ax_main.axis("off")
    else:
        norm_main = make_norm(combined_heat.values.flatten())
        annot_main = combined_heat.map(lambda v: "" if pd.isna(v) else f"{v:.2f}%")
        sns.heatmap(combined_heat, cmap=cmap, norm=norm_main,
                    annot=annot_main, fmt="", linewidths=0.5, linecolor="gray",
                    cbar=False, ax=ax_main)
        ax_main.set_title("Semanal (%) / Retreino Diário (+1) (%)", fontsize=11.0, pad=10)
        ax_main.set_xlabel("")
        ax_main.set_ylabel("Janela (dias)")
        plt.setp(ax_main.get_xticklabels(), rotation=0, ha="center")

    # Heatmap diferenças
    ax_diff = axes[2]
    if combined_heat.empty:
        ax_diff.axis("off")
    else:
        btc_map_numeric = {}
        for col in combined_heat.columns:
            if is_semanal_label(col):
                wd_internal = extract_wd_from_label(col)
                btc_map_numeric[col] = btc_semanal_numeric.get(wd_internal, np.nan)
            elif col == "Retreino Diário (+1)":
                btc_map_numeric[col] = ret_diario_pct_numeric
            else:
                btc_map_numeric[col] = np.nan
        diff_heat = combined_heat.copy()
        for col in diff_heat.columns:
            diff_heat[col] = diff_heat[col] - btc_map_numeric.get(col, np.nan)
        norm_diff = make_norm(diff_heat.values.flatten())
        annot_diff = diff_heat.map(lambda v: "" if pd.isna(v) else f"{v:.2f}%")
        sns.heatmap(diff_heat, cmap=cmap, norm=norm_diff,
                    annot=annot_diff, fmt="", linewidths=0.5, linecolor="gray",
                    cbar=False, ax=ax_diff)
        ax_diff.set_title("Semanal / Retreino Diário (+1) − Valorização BTC", fontsize=10.5, pad=8)
        ax_diff.set_xlabel("")
        ax_diff.set_ylabel("Janela (dias)")
        plt.setp(ax_diff.get_xticklabels(), rotation=0, ha="center")

    plt.tight_layout(rect=[0, 0.0, 1, 0.99])
    fig_path = os.path.join(out_dir, f"{label}_semanais_retreino_diario_multilinha.png")
    plt.savefig(fig_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] {label}: Figura salva: {fig_path}")
    print(f"[INFO] {label}: início={validation_start.date()} fim={validation_end.date()} corte_semana={(validation_end - pd.Timedelta(days=7)).date()}")

def build_periods():
    # Gerar apenas o período de 8 anos (2018-2025)
    return [(pd.Timestamp("2018-01-01"), pd.Timestamp("2025-12-31"), "2018_2025")]

def main():
    input_csv = os.path.join("1-ARIMA", "predictions.csv")
    if not os.path.isfile(input_csv):
        print(f"[ERROR] Arquivo não encontrado: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    base_required = ["date", "dow", "window_days"]
    missing = [c for c in base_required if c not in df.columns]
    if missing:
        print(f"[ERROR] Colunas básicas ausentes: {missing}")
        sys.exit(1)

    daily_horizons = detect_horizons(df.columns, "return_daily_")
    retrain_horizons = detect_horizons(df.columns, "return_retrain_")
    if not daily_horizons:
        print("[ERROR] Nenhum horizonte return_daily_t+h encontrado."); sys.exit(1)
    if not retrain_horizons:
        print("[ERROR] Nenhum horizonte return_retrain_t+h encontrado."); sys.exit(1)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dow"] = pd.to_numeric(df["dow"], errors="coerce")
    df["window_days"] = pd.to_numeric(df["window_days"], errors="coerce")
    df = df.dropna(subset=["date", "dow", "window_days"])

    out_dir = "2-HEATMAPS"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "CSV"), exist_ok=True)

    for s, e, label in build_periods():
        print(f"[INFO] Renderizando período {label}: {s.date()}..{e.date()}")
        render_period(df, daily_horizons, retrain_horizons, s, e, label, out_dir)

    print("[INFO] Concluído.")

if __name__ == "__main__":
    main()
