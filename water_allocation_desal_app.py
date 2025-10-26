import re
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, timedelta, datetime
from scipy.optimize import minimize

# =========================================================
# 💧 Water Allocation + Desalination Web App (Streamlit)
#   • Interval-aware Bois d'Arc cap (annual average in MGD)
#   • User-selectable desal policy with per-facility floors
#   • Ratio modes: Manual / Capacity-based / Optimized (with avg-MGD floors)
#   • Reference SCADA picked by YEAR (now supports CSV or Excel with multi-sheet)
#   • Leap-year-safe alignment of the selected year to the Projection Year
#   • Results table shown after plots; plots selectable without recompute
#   • KPIs: Peaks + Averages for key series
#   • UPDATE: Plant capacities moved to left sidebar (Model Parameters)
# =========================================================

st.set_page_config(page_title="Water Allocation + Desal", layout="wide")

# ------------ Defaults used by the optimizer to enforce capacities ------------
DEFAULT_WYLIE_CAP     = 830.0
DEFAULT_LEONARD_CAP   = 280.0
DEFAULT_TAWAKONI_CAP  = 220.0

# -------------------------------
# Utilities (dates, parsing, alignment)
# -------------------------------
def is_leap_year(y: int) -> bool:
    return calendar.isleap(y)

def generate_daily_dates(year: int):
    start, end = date(year, 1, 1), date(year, 12, 31)
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]

def build_target_dates(year: int):
    start, end = date(year, 1, 1), date(year, 12, 31)
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]

def assert_scada_length(scada_series, expected_len):
    if len(scada_series) != expected_len:
        st.error(
            f"❌ SCADA data length ({len(scada_series)}) does not match the number of days "
            f"for the chosen year ({expected_len}). Please provide exactly {expected_len} daily values in the selected column."
        )
        st.stop()

def make_interval_masks(dates, start_str, end_str, include_end=False):
    """Return boolean masks (inside, outside) for the interval [start, end) or [start, end] if include_end."""
    start = datetime.strptime(start_str, "%m/%d/%Y")
    end   = datetime.strptime(end_str,   "%m/%d/%Y")
    if include_end:
        end = end + pd.Timedelta(days=1)
    d = pd.to_datetime(dates)
    inside  = (d >= start) & (d < end)
    outside = ~inside
    return np.asarray(inside, dtype=bool), np.asarray(outside, dtype=bool)

def parse_efficiency_text(text: str) -> float:
    """Parse a numeric efficiency from text. Accepts decimals (e.g., 0.85) or fractions like '50/59'."""
    s = (text or "").strip()
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            val = float(a) / float(b)
        else:
            val = float(s)
    except Exception:
        st.error("❌ Enter a number or a fraction like '50/59'.")
        st.stop()
    if not np.isfinite(val) or val < 0 or val > 1:
        st.error("❌ Desalination efficiency must be between 0 and 1.")
        st.stop()
    return float(val)

def _normalize_monthday_col(s: pd.Series) -> pd.Series:
    """
    Accepts strings like '1/1', '01/01', '01-01', or datetime-like.
    Returns 'MM-DD' strings.
    """
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.strftime("%m-%d")

def _coerce_numeric(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        raise ValueError("Selected year column is non-numeric.")
    return s

def align_series_to_projection_year(values: pd.Series,
                                    projection_year: int,
                                    monthday: pd.Series | None = None) -> pd.Series:
    """
    Return a Series aligned to the projection year (365/366 days).
    - If 'monthday' is provided (MM-DD), map by month/day.
    - Otherwise, assume Jan1..Dec31 order and fix leap-day differences by inserting/dropping Feb-29.
    """
    tgt_dates = pd.to_datetime(build_target_dates(projection_year))
    tgt_mmdd = tgt_dates.strftime("%m-%d")

    if monthday is not None:
        mmdd = monthday.astype(str)
        df = pd.DataFrame({"MMDD": mmdd, "VAL": values})
        df = df.dropna(subset=["VAL"]).copy()
        df = df.groupby("MMDD", as_index=False)["VAL"].mean()
        aligned = pd.DataFrame({"MMDD": tgt_mmdd}).merge(df, on="MMDD", how="left")["VAL"]

        if "02-29" in set(tgt_mmdd) and aligned.isna().any():
            idx = np.where(tgt_mmdd == "02-29")[0]
            if len(idx) == 1 and pd.isna(aligned.iloc[idx[0]]):
                feb28 = aligned.iloc[idx[0]-1] if idx[0] > 0 else np.nan
                mar01 = aligned.iloc[idx[0]+1] if idx[0] + 1 < len(aligned) else np.nan
                if pd.notna(feb28) and pd.notna(mar01):
                    aligned.iloc[idx[0]] = 0.5 * (feb28 + mar01)
                elif pd.notna(feb28):
                    aligned.iloc[idx[0]] = feb28
                elif pd.notna(mar01):
                    aligned.iloc[idx[0]] = mar01

        return aligned.astype(float).reset_index(drop=True)

    vals = pd.to_numeric(values, errors="coerce").astype(float).reset_index(drop=True)
    tgt_len = len(tgt_mmdd)
    src_len = len(vals)

    if src_len == tgt_len:
        return vals

    if tgt_len == 365 and src_len == 366:
        return vals.drop(vals.index[59]).reset_index(drop=True)

    if tgt_len == 366 and src_len == 365:
        feb28 = vals.iloc[58] if 58 < len(vals) else np.nan
        mar01 = vals.iloc[59] if 59 < len(vals) else np.nan
        if pd.notna(feb28) and pd.notna(mar01):
            feb29 = 0.5 * (feb28 + mar01)
        else:
            feb29 = feb28 if pd.notna(feb28) else mar01
        before = vals.iloc[:59]
        after  = vals.iloc[59:]
        return pd.concat([before, pd.Series([feb29]), after], ignore_index=True)

    raise ValueError(f"Length mismatch: source={src_len}, target={tgt_len}. "
                     "Provide a Date/MonthDay column or ensure daily sequence covers the full year.")

# -------------------------------
# Leonard → Wylie rebalancing to meet annual Bois d'Arc average limit
# -------------------------------
def rebalance_leonard_to_wylie(
    Wylie_D,
    Leonard_D,
    mask,
    mix_ratio_to_leonard: float,
    boisd_avg_limit_mgd: float,
    num_days: int,
):
    """
    Enforce annual Bois d'Arc average limit B (MGD) via implied cap on Leonard total MG:
      MaxLeonardTotal = ((m+1)/m) * B * N
    If Leonard total exceeds this cap, reduce Leonard on the mask days and add the
    exact amounts to Wylie (mass-conservative). If insufficient room on mask days,
    reduce as much as possible (feasible=False).
    """
    m = float(mix_ratio_to_leonard)
    B = float(boisd_avg_limit_mgd)

    max_leonard_total = ((m + 1.0) / m) * B * num_days

    total_leonard = float(np.sum(Leonard_D))
    if total_leonard <= max_leonard_total:
        return Wylie_D, Leonard_D, True

    excess = total_leonard - max_leonard_total

    cap_per_day = Leonard_D[mask].copy()
    total_cap = float(np.sum(cap_per_day))

    W_new = Wylie_D.copy()
    L_new = Leonard_D.copy()

    if total_cap <= 1e-12:
        return W_new, L_new, False

    if total_cap <= excess + 1e-9:
        L_new[mask] = 0.0
        W_new[mask] = W_new[mask] + cap_per_day
        return W_new, L_new, False

    weights = cap_per_day / total_cap
    reduction = weights * excess
    reduction = np.minimum(reduction, cap_per_day)

    L_new[mask] = L_new[mask] - reduction
    W_new[mask] = W_new[mask] + reduction

    return W_new, L_new, True

# -------------------------------
# Desal policies (always use full daily Desal), per-facility floors
# -------------------------------
def desal_update_taw_first(Wylie_D, Tawakoni_D, Desal, floor_taw=5.0):
    """Desal offsets Tawakoni first: pull from Taw down to floor, remainder from Wylie (down to 0)."""
    Taw   = np.asarray(Tawakoni_D, dtype=float)
    Wly   = np.asarray(Wylie_D,    dtype=float)
    Desal = np.asarray(Desal,      dtype=float)

    take_taw  = np.minimum(np.maximum(Taw - floor_taw, 0.0), Desal)
    remainder = Desal - take_taw

    New_Taw = Taw - take_taw
    New_Wly = np.maximum(Wly - remainder, 0.0)
    return New_Wly, New_Taw

def desal_update_wly_first(Wylie_D, Tawakoni_D, Desal, floor_wly=5.0):
    """Desal offsets Wylie first: pull from Wylie down to floor, remainder from Taw (down to 0)."""
    Taw   = np.asarray(Tawakoni_D, dtype=float)
    Wly   = np.asarray(Wylie_D,    dtype=float)
    Desal = np.asarray(Desal,      dtype=float)

    take_wly  = np.minimum(np.maximum(Wly - floor_wly, 0.0), Desal)
    remainder = Desal - take_wly

    New_Wly = Wly - take_wly
    New_Taw = np.maximum(Taw - remainder, 0.0)
    return New_Wly, New_Taw

# -------------------------------
# Core calculations with desal + interval-aware Bois d'Arc cap
# -------------------------------
def compute_demands_desal(
    rW, rL, rT,
    Year,
    SCADA_Data,
    Peak_Day_Demand,
    Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
    Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
    Max_Avg_From_Bois_DARC,
    interval_start_str, interval_end_str, shift_where: str, include_end: bool,
    desal_policy: str, floor_wylie: float, floor_taw: float, desal_efficiency: float,
    enforce_daily_caps: bool,
    Wylie_Cap: float = None, Leonard_Cap: float = None, Tawakoni_Cap: float = None,
):
    dates = generate_daily_dates(Year)
    assert_scada_length(SCADA_Data, len(dates))

    SCADA = np.asarray(SCADA_Data, dtype=float)

    Wylie_Peak    = rW * Peak_Day_Demand
    Leonard_Peak  = rL * Peak_Day_Demand
    Tawakoni_Peak = rT * Peak_Day_Demand

    ref_peak = float(np.max(SCADA))
    if ref_peak <= 0:
        st.error("❌ SCADA peak must be positive.")
        st.stop()

    Wylie_pre    = (Wylie_Peak    / ref_peak) * SCADA
    Leonard_pre  = (Leonard_Peak  / ref_peak) * SCADA
    Tawakoni_pre = (Tawakoni_Peak / ref_peak) * SCADA

    Wylie_D, Leonard_D, Tawakoni_D = Wylie_pre.copy(), Leonard_pre.copy(), Tawakoni_pre.copy()

    cap_report = None

    if enforce_daily_caps:
        if Wylie_Cap is None or Leonard_Cap is None or Tawakoni_Cap is None:
            st.error("❌ Please provide all three plant caps when caps are enforced.")
            st.stop()

        Wylie_excess    = np.maximum(Wylie_pre    - float(Wylie_Cap),    0.0)
        Leonard_excess  = np.maximum(Leonard_pre  - float(Leonard_Cap),  0.0)
        Tawakoni_excess = np.maximum(Tawakoni_pre - float(Tawakoni_Cap), 0.0)

        Wylie_D    = np.minimum(Wylie_pre,    float(Wylie_Cap))
        Leonard_D  = np.minimum(Leonard_pre,  float(Leonard_Cap))
        Tawakoni_D = np.minimum(Tawakoni_pre, float(Tawakoni_Cap))

        cap_report = {
            "Wylie":   {"days": int((Wylie_excess    > 0).sum()), "curtailed_MG": float(Wylie_excess.sum())},
            "Leonard": {"days": int((Leonard_excess  > 0).sum()), "curtailed_MG": float(Leonard_excess.sum())},
            "Tawakoni":{"days": int((Tawakoni_excess > 0).sum()), "curtailed_MG": float(Tawakoni_excess.sum())},
        }

    inside, outside = make_interval_masks(dates, interval_start_str, interval_end_str, include_end)
    shift_mask = inside if (shift_where == "inside") else outside

    Wylie_D, Leonard_D, feasible_bois = rebalance_leonard_to_wylie(
        Wylie_D=Wylie_D,
        Leonard_D=Leonard_D,
        mask=shift_mask,
        mix_ratio_to_leonard=float(Mix_Ratio_To_Leonard),
        boisd_avg_limit_mgd=float(Max_Avg_From_Bois_DARC),
        num_days=len(dates),
    )

    Texoma_L = np.minimum(Leonard_D / (Mix_Ratio_To_Leonard + 1.0), Pipe_Cap_To_Leonard)
    BoisD_L  = Leonard_D - Texoma_L

    df2 = pd.DataFrame({
        "Date": dates,
        "Wylie_D": Wylie_D,
        "Leonard_D": Leonard_D,
        "Texoma_L": Texoma_L,
        "BoisD_L": BoisD_L
    })
    df2.loc[df2["Texoma_L"] == Pipe_Cap_To_Leonard, "BoisD_L"] = (float(Mix_Ratio_To_Leonard) * df2["Texoma_L"])
    df2["Leonard_D"] = df2["Texoma_L"] + df2["BoisD_L"]
    difference = Leonard_D - df2["Leonard_D"].values
    df2["Wylie_D"] = df2["Wylie_D"] + difference

    Wylie_D  = df2["Wylie_D"].values
    Leonard_D = df2["Leonard_D"].values
    BoisD_L   = df2["BoisD_L"].values
    Texoma_L  = df2["Texoma_L"].values

    remaining_pipe_headroom = np.maximum(Pipe_Cap_To_Leonard - Texoma_L, 0.0)
    Desal = remaining_pipe_headroom * desal_efficiency

    if desal_policy == "none":
        Desal = np.zeros_like(remaining_pipe_headroom)
    elif desal_policy == "taw_first":
        Wylie_D, Tawakoni_D = desal_update_taw_first(Wylie_D, Tawakoni_D, Desal, floor_taw=floor_taw)
    elif desal_policy == "wly_first":
        Wylie_D, Tawakoni_D = desal_update_wly_first(Wylie_D, Tawakoni_D, Desal, floor_wly=floor_wylie)
    else:
        Desal = np.zeros_like(remaining_pipe_headroom)

    Texoma_W = np.minimum(Wylie_D / (Mix_Ratio_To_Wylie + 1.0), Pipe_Cap_To_Wylie)
    Lavon_W  = Wylie_D - Texoma_W

    Total_From_Tex = Texoma_W + Texoma_L + Desal
    Total_Demand   = Texoma_W + Lavon_W + Texoma_L + BoisD_L + Tawakoni_D + Desal

    for name, arr in [("Wylie_D", Wylie_D), ("Leonard_D", Leonard_D), ("Tawakoni_D", Tawakoni_D), ("Desal", Desal)]:
        if np.any(np.asarray(arr) < -1e-9):
            st.error(f"❌ {name} went negative. Check policy/floors.")
            st.stop()

    achieved_bois_avg = float(np.mean(BoisD_L))

    return (
        dates, SCADA,
        Wylie_D, Texoma_W, Lavon_W,
        Leonard_D, Texoma_L, BoisD_L,
        Tawakoni_D, Desal,
        Total_From_Tex, Total_Demand,
        feasible_bois, achieved_bois_avg,
        cap_report,
    )

# -------------------------------
# Optimization helpers (enforce caps + avg-MGD floors)
# -------------------------------
def _simulate_with_ratios(
    r, Year, SCADA_Data, Peak_Day_Demand,
    Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
    Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
    Max_Avg_From_Bois_DARC,
    interval_start_str, interval_end_str, shift_where, include_end,
    desal_policy, floor_wylie, floor_taw, desal_efficiency,
    Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
):
    return compute_demands_desal(
        rW=r[0], rL=r[1], rT=r[2],
        Year=Year,
        SCADA_Data=SCADA_Data,
        Peak_Day_Demand=Peak_Day_Demand,
        Mix_Ratio_To_Wylie=Mix_Ratio_To_Wylie,
        Mix_Ratio_To_Leonard=Mix_Ratio_To_Leonard,
        Pipe_Cap_To_Wylie=Pipe_Cap_To_Wylie,
        Pipe_Cap_To_Leonard=Pipe_Cap_To_Leonard,
        Max_Avg_From_Bois_DARC=Max_Avg_From_Bois_DARC,
        interval_start_str=interval_start_str,
        interval_end_str=interval_end_str,
        shift_where=shift_where, include_end=include_end,
        desal_policy=desal_policy, floor_wylie=floor_wylie, floor_taw=floor_taw,
        desal_efficiency=desal_efficiency,
        enforce_daily_caps=True,
        Wylie_Cap=Wylie_Cap_opt, Leonard_Cap=Leonard_Cap_opt, Tawakoni_Cap=Tawakoni_Cap_opt,
    )

def _objective_max_texoma_with_desal(
    r,
    Year, SCADA_Data, Peak_Day_Demand,
    Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
    Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
    Max_Avg_From_Bois_DARC,
    interval_start_str, interval_end_str, shift_where, include_end,
    desal_policy, floor_wylie, floor_taw, desal_efficiency,
    Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
):
    if np.any(r < -1e-8) or abs(np.sum(r) - 1.0) > 1e-6:
        return 1e12
    (_, _, _, _, _, _, _, _, _, _, Total_From_Tex, _, _, _, _) = _simulate_with_ratios(
        r, Year, SCADA_Data, Peak_Day_Demand,
        Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
        Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
        Max_Avg_From_Bois_DARC,
        interval_start_str, interval_end_str, shift_where, include_end,
        desal_policy, floor_wylie, floor_taw, desal_efficiency,
        Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
    )
    return -float(np.sum(Total_From_Tex))

def _avg_mgd_constraint(which,
    r,
    Year, SCADA_Data, Peak_Day_Demand,
    Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
    Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
    Max_Avg_From_Bois_DARC,
    interval_start_str, interval_end_str, shift_where, include_end,
    desal_policy, floor_wylie, floor_taw, desal_efficiency,
    Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
    floor_mgd,
):
    (_dates, _SCADA,
     Wylie_D, Texoma_W, Lavon_W,
     Leonard_D, Texoma_L, BoisD_L,
     Tawakoni_D, Desal,
     _Total_From_Tex, _Total_Demand,
     _feasible_bois, _achieved_bois_avg,
     _cap_report) = _simulate_with_ratios(
        r, Year, SCADA_Data, Peak_Day_Demand,
        Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
        Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
        Max_Avg_From_Bois_DARC,
        interval_start_str, interval_end_str, shift_where, include_end,
        desal_policy, floor_wylie, floor_taw, desal_efficiency,
        Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
    )
    avg = {"W": np.mean(Wylie_D), "L": np.mean(Leonard_D), "T": np.mean(Tawakoni_D)}[which]
    return float(avg - floor_mgd)

def optimize_ratios_with_desal(
    Year, SCADA_Data, Peak_Day_Demand,
    Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
    Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
    Max_Avg_From_Bois_DARC,
    interval_start_str, interval_end_str, shift_where, include_end,
    desal_policy, floor_wylie, floor_taw, desal_efficiency,
    Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
    Wylie_floor_opt, Leonard_floor_opt, Tawakoni_floor_opt,
):
    """
    Multi-start SLSQP:
      - Tries several x0 seeds (capacity-based, equal, W-heavy, L-heavy, WL-heavy, T-heavy)
      - Keeps the solution with the best (lowest) objective value (objective is -Total_From_Tex)
      - Everything else (constraints, desal/rebalance logic) stays the same
    """
    if Peak_Day_Demand <= 0:
        st.error("❌ Peak Day Demand must be > 0.")
        st.stop()

    # --- Compute peak-based ratio limits ---
    rmax_W = min(1.0, Wylie_Cap_opt / Peak_Day_Demand)
    rmax_L = min(1.0, Leonard_Cap_opt / Peak_Day_Demand)
    rmax_T = min(1.0, Tawakoni_Cap_opt / Peak_Day_Demand)
    rmax = np.array([rmax_W, rmax_L, rmax_T], dtype=float)

    # --- Feasibility check ---
    if (rmax_W + rmax_L + rmax_T) < 1.0 - 1e-6:
        st.error(
            f"❌ Infeasible setup:\n"
            f"Total capacity ({Wylie_Cap_opt + Leonard_Cap_opt + Tawakoni_Cap_opt:.1f} MGD) "
            f"is insufficient to meet Peak Day Demand ({Peak_Day_Demand:.1f} MGD)."
        )
        st.stop()

    # Helper: clip to bounds and renormalize to sum=1
    def _clip_and_renorm(r, rmax_vec):
        r = np.asarray(r, dtype=float)
        r = np.maximum(r, 0.0)
        r = np.minimum(r, rmax_vec)
        s = r.sum()
        if s <= 0:
            # fallback: capacity-based
            if rmax_vec.sum() <= 0:
                return np.array([1/3, 1/3, 1/3], dtype=float)
            r = rmax_vec / rmax_vec.sum()
        else:
            r = r / s
        # final safety bound
        r = np.minimum(np.maximum(r, 0.0), rmax_vec)
        r = r / max(r.sum(), 1e-12)
        return r

    # --- Seed set (all get clipped+renormed) ---
    seeds_raw = [
        rmax / rmax.sum(),                                  # capacity-based
        np.array([1/3, 1/3, 1/3], dtype=float),             # equal
        np.array([0.80, 0.15, 0.05], dtype=float),          # Wylie-heavy
        np.array([0.15, 0.80, 0.05], dtype=float),          # Leonard-heavy
        np.array([0.45, 0.45, 0.10], dtype=float),          # Wylie+Leonard heavy
        np.array([0.05, 0.05, 0.90], dtype=float),          # Tawakoni-heavy (for completeness)
    ]
    seeds = [_clip_and_renorm(s, rmax) for s in seeds_raw]

    # --- Common bounds & constraints for SLSQP ---
    bounds = [(0.0, rmax_W), (0.0, rmax_L), (0.0, rmax_T)]
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    cons += [
        {"type": "ineq", "fun": lambda r: _avg_mgd_constraint("W", r,
            Year, SCADA_Data, Peak_Day_Demand,
            Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
            Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
            Max_Avg_From_Bois_DARC,
            interval_start_str, interval_end_str, shift_where, include_end,
            desal_policy, floor_wylie, floor_taw, desal_efficiency,
            Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
            Wylie_floor_opt,
        )},
        {"type": "ineq", "fun": lambda r: _avg_mgd_constraint("L", r,
            Year, SCADA_Data, Peak_Day_Demand,
            Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
            Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
            Max_Avg_From_Bois_DARC,
            interval_start_str, interval_end_str, shift_where, include_end,
            desal_policy, floor_wylie, floor_taw, desal_efficiency,
            Leonard_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
            Leonard_floor_opt,
        )},
        {"type": "ineq", "fun": lambda r: _avg_mgd_constraint("T", r,
            Year, SCADA_Data, Peak_Day_Demand,
            Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
            Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
            Max_Avg_From_Bois_DARC,
            interval_start_str, interval_end_str, shift_where, include_end,
            desal_policy, floor_wylie, floor_taw, desal_efficiency,
            Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
            Tawakoni_floor_opt,
        )},
    ]

    # --- Try each seed; keep the best by objective value (minimize objective = maximize Texoma) ---
    best_r = None
    best_obj = np.inf
    best_success = False
    best_seed_idx = None

    for idx, x0 in enumerate(seeds):
        try:
            res = minimize(
                _objective_max_texoma_with_desal,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                args=(
                    Year, SCADA_Data, Peak_Day_Demand,
                    Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
                    Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
                    Max_Avg_From_Bois_DARC,
                    interval_start_str, interval_end_str, shift_where, include_end,
                    desal_policy, floor_wylie, floor_taw, desal_efficiency,
                    Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
                ),
                options={"maxiter": 400, "ftol": 1e-9}
            )
            cand = res.x if res.success else x0
            cand = _clip_and_renorm(cand, rmax)
            obj = _objective_max_texoma_with_desal(
                cand,
                Year, SCADA_Data, Peak_Day_Demand,
                Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
                Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
                Max_Avg_From_Bois_DARC,
                interval_start_str, interval_end_str, shift_where, include_end,
                desal_policy, floor_wylie, floor_taw, desal_efficiency,
                Wylie_Cap_opt, Leonard_Cap_opt, Tawakoni_Cap_opt,
            )
            if obj < best_obj:
                best_obj = obj
                best_r = cand
                best_success = res.success
                best_seed_idx = idx
        except Exception:
            # Ignore this seed on any numerical error and continue
            continue

    # Fallback: if nothing worked (extremely unlikely), use capacity-based
    if best_r is None:
        best_r = _clip_and_renorm(rmax / rmax.sum(), rmax)
        best_success = False
        best_seed_idx = -1

    r = np.clip(best_r, 0.0, 1.0)
    r = r / max(r.sum(), 1e-12)

    st.success(
        f"Optimized ratios (multi-start) → "
        f"Wylie={r[0]:.3f} (≤ {rmax_W:.3f}), "
        f"Leonard={r[1]:.3f} (≤ {rmax_L:.3f}), "
        f"Tawakoni={r[2]:.3f} (≤ {rmax_T:.3f})"
    )
    # Optional breadcrumb about which seed won (comment out if you prefer quieter UI)
    seed_names = ["capacity", "equal", "W-heavy", "L-heavy", "WL-heavy", "T-heavy"]
    try:
        if best_seed_idx is not None and 0 <= best_seed_idx < len(seed_names):
            st.caption(f"Best run started from: **{seed_names[best_seed_idx]}** seed "
                       f"({'converged' if best_success else 'fallback to seed'})")
    except Exception:
        pass

    return r

# -------------------------------
# Plot helpers (with value/percent annotations)
# -------------------------------
def _plot_series(dates, series, title, ylabel, color, extra_line=None):
    avg, total = float(np.mean(series)), float(np.sum(series))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, series, label="DAILY DATA", color=color)
    ax.axhline(y=avg, color="red", linestyle="--", label=f"AVERAGE = {avg:.1f} (MGD)")
    ax.plot([], [], ' ', label=f"SUM = {total:.1f} (MG)")
    if extra_line is not None:
        ax.axhline(y=extra_line, color="green", linestyle=":", label=f"Target = {extra_line} MGD")
    ax.set_title(title); ax.set_xlabel("DATE"); ax.set_ylabel(ylabel)
    ax.legend(); fig.tight_layout()
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    return fig

def _month_labels():
    return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _monthly_group_average(dates, series):
    """Return 12-length vector of monthly averages aligned Jan..Dec."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "val": np.asarray(series, dtype=float)})
    df["m"] = df["date"].dt.month
    return df.groupby("m")["val"].mean().reindex(range(1,13)).values

def _bar_monthly(dates, series, title, ylabel):
    """Single series monthly average bar with value labels."""
    vals = _monthly_group_average(dates, series)
    labels = _month_labels()
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(np.arange(1,13), vals, tick_label=labels)
    ax.set_title(title); ax.set_xlabel("MONTH"); ax.set_ylabel(ylabel)
    for i, v in enumerate(vals, start=1):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig

def _grouped_monthly_three(dates, s1, s2, s3, labels_legend, title, ylabel):
    """
    Grouped bars (3 per month).  **No numeric value labels on bars** per request.
    """
    m1 = _monthly_group_average(dates, s1)
    m2 = _monthly_group_average(dates, s2)
    m3 = _monthly_group_average(dates, s3)
    labels = _month_labels()
    x = np.arange(12)
    width = 0.26
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x - width, m1, width, label=labels_legend[0])
    ax.bar(x,         m2, width, label=labels_legend[1])
    ax.bar(x + width, m3, width, label=labels_legend[2])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(title); ax.set_xlabel("MONTH"); ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig

def _stacked_monthly_percent(dates, series_dict, title, ylabel):
    """
    Stacked bars of monthly averages with percentage labels on each segment.
    series_dict: Ordered {label: array-like}, typically two series for source mixes.
    """
    labels = _month_labels()
    months = np.arange(1, 13)
    monthly = {k: _monthly_group_average(dates, v) for k, v in series_dict.items()}
    order = list(series_dict.keys())

    fig, ax = plt.subplots(figsize=(12,6))
    bottom = np.zeros(12)
    bars_by_label = {}

    for label in order:
        vals = np.asarray(monthly[label], dtype=float)
        bars = ax.bar(months, vals, bottom=bottom, label=label)
        bars_by_label[label] = (bars, vals.copy(), bottom.copy())
        bottom += vals

    # percentage annotations
    totals = bottom
    for label in order:
        bars, vals, bottoms = bars_by_label[label]
        for i, rect in enumerate(bars):
            total = totals[i]
            val = vals[i]
            if total > 0 and np.isfinite(val):
                pct = 100.0 * val / total
                y = bottoms[i] + val/2.0
                ax.text(rect.get_x() + rect.get_width()/2, y, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=8, color="black")

    ax.set_title(title); ax.set_xlabel("MONTH"); ax.set_ylabel(ylabel)
    ax.set_xticks(months); ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig

def _cumulative_plot(dates, series, title, ylabel):
    cum = np.cumsum(np.asarray(series, dtype=float))
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(dates, cum)
    ax.set_title(title); ax.set_xlabel("DATE"); ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig

# -------------------------------
# Sidebar: core parameters
# -------------------------------
left, mid, right = st.columns([1,3,1])
with mid:
    st.title("💧 Water Demand Projection Web APP")

st.sidebar.header("Model Parameters")
Year = st.sidebar.number_input("Projection Year", min_value=2000, max_value=2100, value=2050, step=1)
Peak_Day_Demand = st.sidebar.number_input("Peak Day Demand (MGD)", value=1209.0)

st.sidebar.subheader("Plant Capacities (MGD)")
capW_sidebar = st.sidebar.number_input("Wylie capacity (MGD)", value=DEFAULT_WYLIE_CAP)
capL_sidebar = st.sidebar.number_input("Leonard capacity (MGD)", value=DEFAULT_LEONARD_CAP)
capT_sidebar = st.sidebar.number_input("Tawakoni capacity (MGD)", value=DEFAULT_TAWAKONI_CAP)

Mix_Ratio_To_Wylie   = st.sidebar.number_input("Mixing Ratio (Lavon:Texoma)",   value=4.0)
Mix_Ratio_To_Leonard = st.sidebar.number_input("Mixing Ratio (BoisD:Texoma)", value=3.0)

Pipe_Cap_To_Wylie   = st.sidebar.number_input("Pipe Capacity From Texoma to Wylie (MGD)",   value=120.0)
Pipe_Cap_To_Leonard = st.sidebar.number_input("Pipe Capacity From Texoma to Leonard (MGD)", value=70.0)

Max_Avg_From_Bois_DARC = st.sidebar.number_input("Bois d'Arc Annual Average Limit (MGD)", value=82.0)

st.sidebar.subheader("Interval for Leonard→Wylie Shift")
interval_start_str = st.sidebar.text_input("Interval Start (MM/DD/YYYY)", value=f"06/01/{Year}")
interval_end_str   = st.sidebar.text_input("Interval End (MM/DD/YYYY)",   value=f"10/01/{Year}")
include_end        = st.sidebar.checkbox("Include End Day in Interval", value=False)
shift_where        = st.sidebar.radio("Shift occurs on:", ["outside", "inside"], horizontal=True)

st.sidebar.subheader("Desalination")
desal_policy = st.sidebar.radio(
    "Desal policy",
    ["none", "taw_first", "wly_first"],
    format_func=lambda s: {
        "none": "No desalination",
        "taw_first": "Desal offsets Tawakoni first",
        "wly_first": "Desal offsets Wylie first",
    }[s],
)
_desal_eff_text = st.sidebar.text_input(
    "Desalination efficiency (Desaled Water/Raw Water)",
    value="50/59",
    help="You can type a decimal like 0.85 or a fraction like 50/59",
)
desal_efficiency = parse_efficiency_text(_desal_eff_text)
floor_wylie = st.sidebar.number_input("Wylie floor (MGD)", value=5.0, min_value=0.0)
floor_taw   = st.sidebar.number_input("Tawakoni floor (MGD)", value=5.0, min_value=0.0)

# -------------------------------
# Pumpage upload (CSV or Excel), sheet picker, year selection, leap-year alignment
# -------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Pumpage file (CSV or Excel) — columns per year (e.g., 2011, 2012, ...)",
    type=["csv", "xlsx", "xls"]
)
if not uploaded_file:
    st.warning("⚠️ Please upload a pumpage file to continue.")
    st.stop()

sheet_df = None
monthday_series = None

try:
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Choose sheet", options=xls.sheet_names)
        sheet_df = pd.read_excel(xls, sheet_name=sheet_name)
    else:
        sheet_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Could not read file: {e}")
    st.stop()

if sheet_df is None or sheet_df.empty:
    st.error("❌ Uploaded file is empty.")
    st.stop()

date_like_cols = [c for c in sheet_df.columns if str(c).strip().lower() in ["date", "monthday", "month_day", "mmdd", "month-day"]]
if date_like_cols:
    _dlc = date_like_cols[0]
    monthday_series = _normalize_monthday_col(sheet_df[_dlc])

year_cols = [c for c in sheet_df.columns if re.fullmatch(r"\d{4}", str(c).strip())]
if not year_cols:
    tmp = []
    for c in sheet_df.columns:
        s = re.sub(r"[^\d]", "", str(c))
        if len(s) == 4:
            tmp.append(c)
    year_cols = tmp

if not year_cols:
    st.error("❌ No year columns found. Make sure your columns are named like 2011, 2012, ...")
    st.stop()

year_cols_sorted = sorted(year_cols, key=lambda x: int(re.sub(r"[^\d]", "", str(x))))
ref_year = st.sidebar.selectbox("Reference Year (pick a column)", options=year_cols_sorted, index=len(year_cols_sorted)-1)

try:
    ref_col = _coerce_numeric(sheet_df[ref_year])
except Exception as e:
    st.error(f"❌ Problem with selected column: {e}")
    st.stop()

try:
    SCADA_Data_col = align_series_to_projection_year(ref_col, Year, monthday=monthday_series)
except Exception as e:
    st.error(f"❌ Could not align the selected year to {Year}: {e}")
    st.stop()

expected_len = 366 if is_leap_year(Year) else 365
if len(SCADA_Data_col) != expected_len:
    st.error(f"❌ After alignment, series length is {len(SCADA_Data_col)} but expected {expected_len} for {Year}.")
    st.stop()

if pd.isna(SCADA_Data_col).any():
    SCADA_Data_col = SCADA_Data_col.fillna(method="ffill").fillna(method="bfill")

# Keep a clean numeric year for reference-year plotting
ref_year_str = str(ref_year)
ref_year_num = int(re.sub(r"[^\d]", "", ref_year_str)) if re.search(r"\d", ref_year_str) else Year

# Prepare reference-year dates/series for plotting ORIGINAL reference daily demand
ref_expected_len = 366 if is_leap_year(ref_year_num) else 365
ref_dates_for_plot = build_target_dates(ref_year_num)
ref_series_for_plot = pd.to_numeric(ref_col, errors="coerce")

if len(ref_series_for_plot) == ref_expected_len:
    if ref_series_for_plot.isna().any():
        ref_series_for_plot = ref_series_for_plot.fillna(method="ffill").fillna(method="bfill")
    ref_plot_ok = True
else:
    # Fallback: lengths don't match; we will fall back to projection-year alignment when plotting
    ref_plot_ok = False

# -------------------------------
# Optimization floors (average MGD) for each plant
# -------------------------------
with st.sidebar.expander("Optimization: Average MGD Floors (prevent shutdowns)"):
    Wylie_floor_opt    = st.number_input("Wylie avg-MGD floor",   value=10.0, min_value=0.0)
    Leonard_floor_opt  = st.number_input("Leonard avg-MGD floor", value=10.0, min_value=0.0)
    Tawakoni_floor_opt = st.number_input("Tawakoni avg-MGD floor",value=10.0, min_value=0.0)
    st.caption("During optimization, each plant's **average** daily treated flow must be at least these values.")

# -------------------------------
# Ratio Modes
# -------------------------------
mode = st.radio("Select Ratio Mode:", ["Manual", "Capacity-based", "Optimized"], horizontal=True)

if mode == "Manual":
    st.subheader("Manual Ratios (must sum to 1)")
    col1, col2, col3 = st.columns(3)
    with col1:
        rW = st.slider("Wylie Ratio", 0.0, 1.0, 0.65, 0.01)
    with col2:
        rL = st.slider("Leonard Ratio", 0.0, 1.0, 0.2, 0.01)
    with col3:
        rT = st.slider("Tawakoni Ratio", 0.0, 1.0, 0.15, 0.01)
    ratio_sum = rW + rL + rT
    if abs(ratio_sum - 1.0) > 1e-6:
        st.error(f"❌ Ratios must sum to 1. Current sum = {ratio_sum:.2f}")
        st.stop()
    st.info(f"Selected ratios → Wylie={rW:.3f}, Leonard={rL:.3f}, Tawakoni={rT:.3f}")

elif mode == "Capacity-based":
    st.subheader("Capacity-based Ratios")
    total_cap_ratio = capW_sidebar + capL_sidebar + capT_sidebar
    if total_cap_ratio <= 0:
        st.error("❌ Sum of capacities must be > 0 for capacity-based ratios.")
        st.stop()
    rW, rL, rT = capW_sidebar/total_cap_ratio, capL_sidebar/total_cap_ratio, capT_sidebar/total_cap_ratio
    st.info(
        f"Selected ratios (capacity-based) → Wylie={rW:.3f}, Leonard={rL:.3f}, Tawakoni={rT:.3f} "
        f"using caps W={capW_sidebar:.1f}, L={capL_sidebar:.1f}, T={capT_sidebar:.1f} MGD"
    )

else:
    st.subheader("Optimized Ratios (maximize Texoma usage, respect caps, keep all plants on)")
    rW, rL, rT = optimize_ratios_with_desal(
        Year, SCADA_Data_col, Peak_Day_Demand,
        Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
        Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
        Max_Avg_From_Bois_DARC,
        interval_start_str, interval_end_str, shift_where, include_end,
        desal_policy, floor_wylie, floor_taw, desal_efficiency,
        capW_sidebar, capL_sidebar, capT_sidebar,
        Wylie_floor_opt=Wylie_floor_opt, Leonard_floor_opt=Leonard_floor_opt, Tawakoni_floor_opt=Tawakoni_floor_opt,
    )
    st.success(f"Optimized ratios → Wylie={rW:.3f}, Leonard={rL:.3f}, Tawakoni={rT:.3f}")

st.caption(f"Current ratios: Wylie={rW:.3f}, Leonard={rL:.3f}, Tawakoni={rT:.3f}")

# -------------------------------
# NEW: Peaks KPIs
# -------------------------------
ref_peak = float(np.max(SCADA_Data_col))
Wylie_Peak    = rW * Peak_Day_Demand
Leonard_Peak  = rL * Peak_Day_Demand
Tawakoni_Peak = rT * Peak_Day_Demand

pk1, pk2, pk3, pk4 = st.columns(4)
with pk1:
    st.metric("Reference Peak (MGD)", f"{ref_peak:.2f}")
with pk2:
    st.metric("Wylie Peak (MGD)", f"{Wylie_Peak:.2f}")
with pk3:
    st.metric("Leonard Peak (MGD)", f"{Leonard_Peak:.2f}")
with pk4:
    st.metric("Tawakoni Peak (MGD)", f"{Tawakoni_Peak:.2f}")

# -------------------------------
# Session-state handling (compute once, plot many)
# -------------------------------
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'sig' not in st.session_state:
    st.session_state['sig'] = None

_cur_sig = (
    Year, float(Peak_Day_Demand), float(Mix_Ratio_To_Wylie), float(Mix_Ratio_To_Leonard),
    float(Pipe_Cap_To_Wylie), float(Pipe_Cap_To_Leonard), float(Max_Avg_From_Bois_DARC),
    interval_start_str, interval_end_str, include_end, shift_where,
    desal_policy, float(desal_efficiency), float(floor_wylie), float(floor_taw),
    float(rW), float(rL), float(rT), str(ref_year), len(SCADA_Data_col), float(pd.Series(SCADA_Data_col).sum()),
    float(Wylie_floor_opt), float(Leonard_floor_opt), float(Tawakoni_floor_opt),
    float(capW_sidebar), float(capL_sidebar), float(capT_sidebar),
)

params_changed = (st.session_state['sig'] is not None and st.session_state['sig'] != _cur_sig)
if params_changed:
    st.info("Parameters changed since last run — click **Run Analysis** to refresh results.")

run = st.button("Run Analysis", type="primary")
if run:
    (
        dates, SCADA,
        Wylie_D, Texoma_W, Lavon_W,
        Leonard_D, Texoma_L, BoisD_L,
        Tawakoni_D, Desal,
        Total_From_Tex, Total_Demand,
        feasible_bois, achieved_bois_avg,
        cap_report,
    ) = compute_demands_desal(
        rW=rW, rL=rL, rT=rT,
        Year=Year,
        SCADA_Data=SCADA_Data_col,
        Peak_Day_Demand=Peak_Day_Demand,
        Mix_Ratio_To_Wylie=Mix_Ratio_To_Wylie,
        Mix_Ratio_To_Leonard=Mix_Ratio_To_Leonard,
        Pipe_Cap_To_Wylie=Pipe_Cap_To_Wylie,
        Pipe_Cap_To_Leonard=Pipe_Cap_To_Leonard,
        Max_Avg_From_Bois_DARC=Max_Avg_From_Bois_DARC,
        interval_start_str=interval_start_str,
        interval_end_str=interval_end_str,
        shift_where=shift_where, include_end=include_end,
        desal_policy=desal_policy, floor_wylie=floor_wylie, floor_taw=floor_taw,
        desal_efficiency=desal_efficiency,
        enforce_daily_caps=False
    )

    df = pd.DataFrame({
        "DATE": dates,
        "REFERENCE DAILY DEMAND (MGD)": pd.Series(SCADA),
        "WYLIE DAILY DEMAND (MGD)": Wylie_D,
        "TEXOMA TO WYLIE (MGD)": Texoma_W,
        "LAVON TO WYLIE (MGD)": Lavon_W,
        "LEONARD DAILY DEMAND (MGD)": Leonard_D,
        "TEXOMA TO LEONARD (MGD)": Texoma_L,
        "BOIS D ARC TO LEONARD (MGD)": BoisD_L,
        "TAWAKONI DAILY DEMAND (MGD)": Tawakoni_D,
        "DESALINATED WATER (MGD)": Desal,
        "TOTAL FROM TEXOMA (MGD)": Total_From_Tex,
        "TOTAL DEMAND (MGD)": Total_Demand,
    })
    numcols = df.select_dtypes(include=[np.number]).columns
    df[numcols] = df[numcols].round(1)

    # Save ref-year plotting info into session results
    ref_dates_safe = ref_dates_for_plot if ref_plot_ok else dates
    ref_series_safe = ref_series_for_plot.values if ref_plot_ok else np.asarray(SCADA)

    st.session_state['results'] = {
        "df": df,
        "dates": dates,
        "SCADA": SCADA,  # projection-year–aligned reference series (for table and other calcs)
        "ref_dates": ref_dates_safe,      # used ONLY for plotting reference daily demand by ref-year calendar
        "ref_series": ref_series_safe,    # ditto
        "ref_year_num": ref_year_num,

        "Wylie_D": Wylie_D,
        "Texoma_W": Texoma_W,
        "Lavon_W": Lavon_W,
        "Leonard_D": Leonard_D,
        "Texoma_L": Texoma_L,
        "BoisD_L": BoisD_L,
        "Tawakoni_D": Tawakoni_D,
        "Desal": Desal,
        "Total_From_Tex": Total_From_Tex,
        "Total_Demand": Total_Demand,
        "feasible_bois": feasible_bois,
        "achieved_bois_avg": achieved_bois_avg,
        "cap_report": cap_report,
        "sig": _cur_sig,
        "Year": Year,
        "Max_Avg_From_Bois_DARC": Max_Avg_From_Bois_DARC,
        "ref_peak": ref_peak,
        "Wylie_Peak": Wylie_Peak,
        "Leonard_Peak": Leonard_Peak,
        "Tawakoni_Peak": Tawakoni_Peak,
    }
    st.session_state['sig'] = _cur_sig

# -------------------------------
# Render results if available (no recompute needed for plots)
# -------------------------------
if st.session_state['results'] is not None:
    res = st.session_state['results']
    Year_res = res["Year"]

    # KPIs
    k1, k2, k3 , k4 = st.columns(4) 
    with k1:
        st.metric("Annual Avg Wylie WT (MGD)", f"{np.mean(res['Wylie_D']):.1f}")
    with k2:
        st.metric("Annual Avg Leonard WT (MGD)", f"{np.mean(res['Leonard_D']):.1f}")
    with k3:
        st.metric("Annual Avg Tawakoni WT (MGD)", f"{np.mean(res['Tawakoni_D']):.1f}")    
    with k4:
        st.metric("Annual Avg Total Demand (MGD)", f"{np.mean(res['Total_Demand']):.1f}")    

    k5 , k6 , k7, k8 = st.columns(4)
    with k5:
        st.metric("Annual Avg Lavon → Wylie (MGD)", f"{np.mean(res['Lavon_W']):.1f}")
    with k6:
        st.metric("Annual Avg Texoma → Wylie (MGD)", f"{np.mean(res['Texoma_W']):.1f}")
    with k7:
        st.metric("Annual Avg Bois d'Arc → Leonard (MGD)", f"{np.mean(res['BoisD_L']):.1f}",
                  delta=f"Target {res['Max_Avg_From_Bois_DARC']:.1f}")
    with k8:
        st.metric("Annual Avg Texoma → Leonard (MGD)", f"{np.mean(res['Texoma_L']):.1f}")

    k9, k10, k11, k12 = st.columns(4)
    with k9:
        st.metric("Annual Avg Desalination (MGD)", f"{np.mean(res['Desal']):.1f}")
    with k10:
        st.metric("Annual Avg From Texoma (MGD)", f"{np.mean(res['Total_From_Tex']):.1f}")
    with k11:
        st.metric("Sum of WTPs (MGD)", f"{np.mean(res['Wylie_D'])+np.mean(res['Leonard_D'])+np.mean(res['Tawakoni_D'])+np.mean(res['Desal']):.1f}",
                  delta=f"Total Demand {np.mean(res['Total_Demand']):.1f}")
    with k12:
        st.metric("Sum of Lakes (MGD)", f"{np.mean(res['Lavon_W'])+np.mean(res['Texoma_W'])+np.mean(res['BoisD_L'])+np.mean(res['Texoma_L'])+np.mean(res['Tawakoni_D'])+np.mean(res['Desal']):.1f}",
                  delta=f"Total Demand {np.mean(res['Total_Demand']):.1f}")

    # ---------- Unified Plots (Preset + Custom) ----------
    with st.expander("All Plots (Preset & Custom)", expanded=True):
        # --- Preset plots ---
        preset_options = st.multiselect(
            "Preset plots to display:",
            [
                # Existing daily plots
                "Texoma Only",
                "Total Demand",
                "Wylie Demand",
                "Wylie Splits (Texoma vs Lavon)",
                "Leonard Demand",
                "Leonard Splits (Texoma vs BoisD)",
                "Bois D'Arc Flow (with limit line)",
                "Tawakoni Demand",
                "Desalinated Water",
                # Monthly & cumulative views
                "Monthly Avg — Wylie/Leonard/Tawakoni (bar)",
                "Monthly Avg — Total Demand (bar)",
                "Monthly Source Mix → Wylie (stacked %)",
                "Monthly Source Mix → Leonard (stacked %)",
                "Monthly Avg — Desalinated Water (bar)",
                "Cumulative From Texoma (line)",
            ],
            default=[
                "Texoma Only",
                "Total Demand",
                "Bois D'Arc Flow (with limit line)",
                "Desalinated Water",
                "Monthly Avg — Wylie/Leonard/Tawakoni (bar)",
                "Monthly Source Mix → Wylie (stacked %)",
            ],
        )

        # Daily presets
        if "Texoma Only" in preset_options:
            st.pyplot(_plot_series(res['dates'], res['Total_From_Tex'],
                                   f"DAILY WATER FROM TEXOMA ({Year_res})",
                                   "RAW WATER (MGD)", "blue"))
        if "Total Demand" in preset_options:
            st.pyplot(_plot_series(res['dates'], res['Total_Demand'],
                                   f"DAILY TOTAL DEMAND ({Year_res})",
                                   "TOTAL WATER DEMAND (MGD)", "purple"))
        if "Wylie Demand" in preset_options:
            st.pyplot(_plot_series(res['dates'], res['Wylie_D'],
                                   f"WYLIE DEMAND — {Year_res}",
                                   "WYLIE DEMAND (MGD)", "orange"))
        if "Wylie Splits (Texoma vs Lavon)" in preset_options:
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(res['dates'], res['Texoma_W'], label="Texoma → Wylie", color="blue")
            ax.plot(res['dates'], res['Lavon_W'],  label="Lavon → Wylie",  color="gray")
            ax.set_title(f"WYLIE SPLITS ({Year_res})"); ax.set_xlabel("DATE"); ax.set_ylabel("MGD")
            ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.35); fig.tight_layout(); st.pyplot(fig)
        if "Leonard Demand" in preset_options:
            st.pyplot(_plot_series(res['dates'], res['Leonard_D'],
                                   f"LEONARD DEMAND — {Year_res}",
                                   "LEONARD DEMAND (MGD)", "green"))
        if "Leonard Splits (Texoma vs BoisD)" in preset_options:
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(res['dates'], res['Texoma_L'], label="Texoma → Leonard", color="blue")
            ax.plot(res['dates'], res['BoisD_L'],  label="Bois d'Arc → Leonard", color="brown")
            ax.set_title(f"LEONARD SPLITS ({Year_res})"); ax.set_xlabel("DATE"); ax.set_ylabel("MGD")
            ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.35); fig.tight_layout(); st.pyplot(fig)
        if "Bois D'Arc Flow (with limit line)" in preset_options:
            st.pyplot(_plot_series(res['dates'], res['BoisD_L'],
                                   f"BOIS D'ARC → LEONARD — {Year_res}",
                                   "BOIS D'ARC FLOW (MGD)", "brown", extra_line=res['Max_Avg_From_Bois_DARC']))
        if "Tawakoni Demand" in preset_options:
            st.pyplot(_plot_series(res['dates'], res['Tawakoni_D'],
                                   f"TAWAKONI DEMAND — {Year_res}",
                                   "TAWAKONI DEMAND (MGD)", "teal"))
        if "Desalinated Water" in preset_options:
            st.pyplot(_plot_series(res['dates'], res['Desal'],
                                   f"DESALINATED WATER — {Year_res}",
                                   "DESALED FLOW (MGD)", "black"))

        # Monthly/cumulative presets
        if any(opt in preset_options for opt in [
            "Monthly Avg — Wylie/Leonard/Tawakoni (bar)",
            "Monthly Avg — Total Demand (bar)",
            "Monthly Source Mix → Wylie (stacked %)",
            "Monthly Source Mix → Leonard (stacked %)",
            "Monthly Avg — Desalinated Water (bar)",
            "Cumulative From Texoma (line)",
        ]):
            dates_dt = pd.to_datetime(res['dates'])

        if "Monthly Avg — Wylie/Leonard/Tawakoni (bar)" in preset_options:
            st.pyplot(_grouped_monthly_three(
                dates_dt, res['Wylie_D'], res['Leonard_D'], res['Tawakoni_D'],
                labels_legend=("Wylie", "Leonard", "Tawakoni"),
                title=f"MONTHLY AVERAGE — PLANT DEMANDS ({Year_res})",
                ylabel="MGD"
            ))

        if "Monthly Avg — Total Demand (bar)" in preset_options:
            st.pyplot(_bar_monthly(dates_dt, res['Total_Demand'],
                                   f"MONTHLY AVERAGE — TOTAL DEMAND ({Year_res})", "MGD"))

        if "Monthly Source Mix → Wylie (stacked %)" in preset_options:
            st.pyplot(_stacked_monthly_percent(
                dates_dt,
                {"Texoma → Wylie": res['Texoma_W'], "Lavon → Wylie": res['Lavon_W']},
                f"MONTHLY SOURCE MIX → WYLIE ({Year_res})",
                "MGD (monthly average, % labels by segment)"
            ))

        if "Monthly Source Mix → Leonard (stacked %)" in preset_options:
            st.pyplot(_stacked_monthly_percent(
                dates_dt,
                {"Texoma → Leonard": res['Texoma_L'], "Bois d'Arc → Leonard": res['BoisD_L']},
                f"MONTHLY SOURCE MIX → LEONARD ({Year_res})",
                "MGD (monthly average, % labels by segment)"
            ))

        if "Monthly Avg — Desalinated Water (bar)" in preset_options:
            st.pyplot(_bar_monthly(dates_dt, res['Desal'],
                                   f"MONTHLY AVERAGE — DESALINATED WATER ({Year_res})", "MGD"))

        if "Cumulative From Texoma (line)" in preset_options:
            st.pyplot(_cumulative_plot(dates_dt, res['Total_From_Tex'],
                                       f"CUMULATIVE WATER FROM TEXOMA — {Year_res}",
                                       "CUMULATIVE MG"))

        st.markdown("---")
        # --- Custom plots: Time Series & Monthly Bars for ANY result columns ---
        result_df = res['df']
        all_cols = [c for c in result_df.columns if c != "DATE"]
        selected_cols = st.multiselect(
            "Custom plots — choose result columns:",
            options=all_cols,
            default=all_cols[:3] if len(all_cols) >= 3 else all_cols
        )
        plot_kinds = st.multiselect(
            "Choose plot types:",
            options=["Time Series", "Monthly Average Bar"],
            default=["Time Series", "Monthly Average Bar"]
        )

        if selected_cols and plot_kinds:
            dates_dt_all = pd.to_datetime(result_df["DATE"])
            for col in selected_cols:
                # Default to projection-year dates/series
                col_dates = dates_dt_all
                col_series = result_df[col].values
                col_year_label = Year_res

                # Special case: Reference Daily Demand should use reference-year calendar
                if col.strip().upper() == "REFERENCE DAILY DEMAND (MGD)":
                    if "ref_dates" in res and "ref_series" in res and res["ref_dates"] is not None:
                        col_dates = pd.to_datetime(res["ref_dates"])
                        col_series = np.asarray(res["ref_series"], dtype=float)
                        col_year_label = res.get("ref_year_num", Year_res)

                if "Time Series" in plot_kinds:
                    st.pyplot(_plot_series(
                        col_dates, col_series,
                        title=f"{col} — DAILY TIME SERIES ({col_year_label})",
                        ylabel="MGD",
                        color="tab:blue"
                    ))
                if "Monthly Average Bar" in plot_kinds:
                    st.pyplot(_bar_monthly(
                        col_dates, col_series,
                        title=f"MONTHLY AVERAGE — {col} ({col_year_label})",
                        ylabel="MGD"
                    ))
        else:
            st.info("Select at least one column and one plot type to display these charts.")

    # --- Results table AFTER plots ---
    st.subheader("Results (Daily)")
    st.dataframe(res['df'], use_container_width=True, height=420)

    with st.expander("Download results as CSV"):
        file_name = st.text_input("CSV file name", value="WaterResults_Desal.csv")
        st.download_button("📥 Download CSV", res['df'].to_csv(index=False), file_name=file_name, mime="text/csv")

else:
    st.info("Set parameters and click **Run Analysis** to compute results. After that, you can switch plots freely without recomputing.")
