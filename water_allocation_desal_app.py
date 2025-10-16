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
#   • Optional DAILY plant caps with curtailment reporting
#   • Ratio modes: Manual / Capacity-based / Optimized (with avg-MGD floors)
#   • Reference SCADA picked by YEAR (CSV has columns as years)
#   • Results table shown first; plots selectable without recompute
#   • KPIs: Peaks + Averages for key series
# =========================================================

st.set_page_config(page_title="Water Allocation + Desal", layout="wide")

# ------------ Defaults used by the optimizer to enforce capacities ------------
DEFAULT_WYLIE_CAP     = 830.0
DEFAULT_LEONARD_CAP   = 280.0
DEFAULT_TAWAKONI_CAP  = 220.0

# -------------------------------
# Utilities
# -------------------------------
def generate_daily_dates(year: int):
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

    excess = total_leonard - max_leonard_total  # MG to shave from Leonard

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
    # Bois d'Arc average limit (MGD)
    Max_Avg_From_Bois_DARC,
    # Interval + shift side
    interval_start_str, interval_end_str, shift_where: str, include_end: bool,
    # Desal policy + floors + efficiency
    desal_policy: str, floor_wylie: float, floor_taw: float, desal_efficiency: float,
    # Optional daily plant caps
    enforce_daily_caps: bool,
    Wylie_Cap: float = None, Leonard_Cap: float = None, Tawakoni_Cap: float = None,
):
    dates = generate_daily_dates(Year)
    assert_scada_length(SCADA_Data, len(dates))

    SCADA = np.asarray(SCADA_Data, dtype=float)

    # ---------------- Peak per facility from ratios ----------------
    Wylie_Peak    = rW * Peak_Day_Demand
    Leonard_Peak  = rL * Peak_Day_Demand
    Tawakoni_Peak = rT * Peak_Day_Demand

    ref_peak = float(np.max(SCADA))
    if ref_peak <= 0:
        st.error("❌ SCADA peak must be positive.")
        st.stop()

    # Pre-cap treated demands (for cap reporting)
    Wylie_pre    = (Wylie_Peak    / ref_peak) * SCADA
    Leonard_pre  = (Leonard_Peak  / ref_peak) * SCADA
    Tawakoni_pre = (Tawakoni_Peak / ref_peak) * SCADA

    Wylie_D, Leonard_D, Tawakoni_D = Wylie_pre.copy(), Leonard_pre.copy(), Tawakoni_pre.copy()

    cap_report = None

    # ---------------- Optional daily plant caps ----------------
    if enforce_daily_caps:
        if Wylie_Cap is None or Leonard_Cap is None or Tawakoni_Cap is None:
            st.error("❌ Please provide all three plant caps when 'Enforce daily caps' is enabled.")
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

    # ---------------- Interval masks for rebalancing side ----------------
    inside, outside = make_interval_masks(dates, interval_start_str, interval_end_str, include_end)
    shift_mask = inside if (shift_where == "inside") else outside

    # ---------------- Enforce Bois d'Arc annual average via Leonard total cap ----------------
    Wylie_D, Leonard_D, feasible_bois = rebalance_leonard_to_wylie(
        Wylie_D=Wylie_D,
        Leonard_D=Leonard_D,
        mask=shift_mask,
        mix_ratio_to_leonard=float(Mix_Ratio_To_Leonard),
        boisd_avg_limit_mgd=float(Max_Avg_From_Bois_DARC),
        num_days=len(dates),
    )

    # ---------------- Leonard split (after rebalancing) ----------------
    Texoma_L = np.minimum(Leonard_D / (Mix_Ratio_To_Leonard + 1.0), Pipe_Cap_To_Leonard)
    BoisD_L  = Leonard_D - Texoma_L
#############################################################
    # ✅ Correct DataFrame creation 
    df2 = pd.DataFrame({ "Date": dates, "Wylie_D": Wylie_D, "Leonard_D": Leonard_D, "Texoma_L": Texoma_L, "BoisD_L": BoisD_L }) 
    # ✅ Correct conditional assignment 
    df2.loc[df2["Texoma_L"] == Pipe_Cap_To_Leonard, "BoisD_L"] = ( Mix_Ratio_To_Leonard * df2["Texoma_L"] ) 
    df2["Leonard_D"]=df2["Texoma_L"]+df2["BoisD_L"] 
    difference= Leonard_D-df2["Leonard_D"].values 
    df2["Wylie_D"]=Wylie_D+difference 
    Wylie_D=df2["Wylie_D"].values
    Leonard_D=df2["Leonard_D"].values 
    BoisD_L=df2["BoisD_L"].values
###############################################################


    # ---------------- Desal capacity from Leonard pipe remainder ----------------
    # Efficiency: need 59 raw to produce 50 desaled ⇒ efficiency = 50/59
    remaining_pipe_headroom = np.maximum(Pipe_Cap_To_Leonard - Texoma_L, 0.0)
    Desal = remaining_pipe_headroom * desal_efficiency

    # ---------------- Apply desal policy with per-facility floors ----------------
    if desal_policy == "none":
        # No desalination this scenario
        Desal = np.zeros_like(remaining_pipe_headroom)
        # Wylie_D and Tawakoni_D remain unchanged
    elif desal_policy == "taw_first":
        Wylie_D, Tawakoni_D = desal_update_taw_first(Wylie_D, Tawakoni_D, Desal, floor_taw=floor_taw)
    elif desal_policy == "wly_first":
        Wylie_D, Tawakoni_D = desal_update_wly_first(Wylie_D, Tawakoni_D, Desal, floor_wly=floor_wylie)
    else:
        # Fallback: treat as no desal
        Desal = np.zeros_like(remaining_pipe_headroom)

    # ---------------- Wylie split (after desal policy) ----------------
    Texoma_W = np.minimum(Wylie_D / (Mix_Ratio_To_Wylie + 1.0), Pipe_Cap_To_Wylie)
    Lavon_W  = Wylie_D - Texoma_W

    # ---------------- Totals & safety ----------------
    Total_From_Tex = Texoma_W + Texoma_L + Desal  # counts desal as from Texoma pipe
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
    # caps enforced here regardless of UI
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
        enforce_daily_caps=True,  # enforce caps during optimization
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
    # caps for optimization
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
    # maximize Total_From_Tex
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
    return float(avg - floor_mgd)  # >= 0 means OK

def optimize_ratios_with_desal(
    Year, SCADA_Data, Peak_Day_Demand,
    Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
    Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
    Max_Avg_From_Bois_DARC,
    interval_start_str, interval_end_str, shift_where, include_end,
    desal_policy, floor_wylie, floor_taw, desal_efficiency,
    # UI caps (may be None); optimizer will use defaults if None
    enforce_daily_caps, Wylie_Cap, Leonard_Cap, Tawakoni_Cap,
    # avg-MGD floors for optimization
    Wylie_floor_opt, Leonard_floor_opt, Tawakoni_floor_opt,
):
    # Caps used for OPTIMIZATION (always enforced here)
    Wcap_opt = float(Wylie_Cap if Wylie_Cap is not None else DEFAULT_WYLIE_CAP)
    Lcap_opt = float(Leonard_Cap if Leonard_Cap is not None else DEFAULT_LEONARD_CAP)
    Tcap_opt = float(Tawakoni_Cap if Tawakoni_Cap is not None else DEFAULT_TAWAKONI_CAP)

    # Initial guess: capacity-based
    total_cap = Wcap_opt + Lcap_opt + Tcap_opt
    x0 = np.array([Wcap_opt/total_cap, Lcap_opt/total_cap, Tcap_opt/total_cap])

    # Bounds [0,1]
    bounds = [(0.0, 1.0)] * 3

    # Sum-to-one equality
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    # Avg-MGD floors as inequality constraints (fun(x) >= 0)
    cons += [
        {"type": "ineq", "fun": lambda r: _avg_mgd_constraint("W", r,
            Year, SCADA_Data, Peak_Day_Demand,
            Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
            Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
            Max_Avg_From_Bois_DARC,
            interval_start_str, interval_end_str, shift_where, include_end,
            desal_policy, floor_wylie, floor_taw, desal_efficiency,
            Wcap_opt, Lcap_opt, Tcap_opt,
            Wylie_floor_opt,
        )},
        {"type": "ineq", "fun": lambda r: _avg_mgd_constraint("L", r,
            Year, SCADA_Data, Peak_Day_Demand,
            Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
            Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
            Max_Avg_From_Bois_DARC,
            interval_start_str, interval_end_str, shift_where, include_end,
            desal_policy, floor_wylie, floor_taw, desal_efficiency,
            Wcap_opt, Lcap_opt, Tcap_opt,
            Leonard_floor_opt,
        )},
        {"type": "ineq", "fun": lambda r: _avg_mgd_constraint("T", r,
            Year, SCADA_Data, Peak_Day_Demand,
            Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
            Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
            Max_Avg_From_Bois_DARC,
            interval_start_str, interval_end_str, shift_where, include_end,
            desal_policy, floor_wylie, floor_taw, desal_efficiency,
            Wcap_opt, Lcap_opt, Tcap_opt,
            Tawakoni_floor_opt,
        )},
    ]

    res = minimize(
        _objective_max_texoma_with_desal, x0=x0, method="SLSQP", bounds=bounds, constraints=cons,
        args=(
            Year, SCADA_Data, Peak_Day_Demand,
            Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
            Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
            Max_Avg_From_Bois_DARC,
            interval_start_str, interval_end_str, shift_where, include_end,
            desal_policy, floor_wylie, floor_taw, desal_efficiency,
            Wcap_opt, Lcap_opt, Tcap_opt,
        ),
        options={"maxiter": 200}
    )

    # If optimizer fails or violates constraints slightly, clip and renormalize
    r = np.clip(res.x if res.success else x0, 0.0, 1.0)
    s = r.sum()
    if s <= 0:
        r = x0
    else:
        r = r / s
    return r

# -------------------------------
# Plot helpers
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

# -------------------------------
# Sidebar: core parameters
# -------------------------------
left, mid, right = st.columns([1,3,1])
with mid:
    st.title("💧 Water Demand Projection Web APP 💧")

st.sidebar.header("Model Parameters")
Year = st.sidebar.number_input("Projection Year", min_value=2000, max_value=2100, value=2050, step=1)
Peak_Day_Demand = st.sidebar.number_input("Peak Day Demand (MGD)", value=1209.0)

# Mix ratios (default guidance: Wylie Lavon:Texoma=4:1, Leonard BoisD:Texoma=3:1)
Mix_Ratio_To_Wylie   = st.sidebar.number_input("Mixing Ratio (Lavon:Texoma)",   value=4.0)
Mix_Ratio_To_Leonard = st.sidebar.number_input("Mixing Ratio (BoisD:Texoma)", value=3.0)

# Pipe caps
Pipe_Cap_To_Wylie   = st.sidebar.number_input("Pipe Capacity From Texoma to Wylie (MGD)",   value=120.0)
Pipe_Cap_To_Leonard = st.sidebar.number_input("Pipe Capacity From Texoma to Leonard (MGD)", value=70.0)

# Bois d'Arc average limit (MGD)
Max_Avg_From_Bois_DARC = st.sidebar.number_input("Bois d'Arc Annual Average Limit (MGD)", value=82.0)

# Interval controls
st.sidebar.subheader("Interval for Leonard→Wylie Shift")
interval_start_str = st.sidebar.text_input("Interval Start (MM/DD/YYYY)", value=f"06/01/{Year}")
interval_end_str   = st.sidebar.text_input("Interval End (MM/DD/YYYY)",   value=f"10/01/{Year}")
include_end        = st.sidebar.checkbox("Include End Day in Interval", value=False)
shift_where        = st.sidebar.radio("Shift occurs on:", ["outside", "inside"], horizontal=True,
                                      help="Choose whether to reduce Leonard (and add to Wylie) on days inside or outside the selected interval.")

# Desal options
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
# Let user type fractions like 50/59
_desal_eff_text = st.sidebar.text_input(
    "Desalination efficiency (Desaled Water/Raw Water)",
    value="50/59",
    help="You can type a decimal like 0.85 or a fraction like 50/59",
)
desal_efficiency = parse_efficiency_text(_desal_eff_text)
floor_wylie = st.sidebar.number_input("Wylie floor (MGD)", value=5.0, min_value=0.0)
floor_taw   = st.sidebar.number_input("Tawakoni floor (MGD)", value=5.0, min_value=0.0)

# Optional daily plant caps (for the final run/plots; optimizer enforces its own caps regardless)
st.sidebar.subheader("Daily Plant Caps (optional)")
st.sidebar.caption("If enabled, **daily** treated demand at each plant is capped. Excess above the cap is **curtailed** (not reallocated). You'll get a summary of any curtailments.")
enforce_daily_caps = st.sidebar.checkbox("Enforce daily plant caps", value=False)
Wylie_Cap = Leonard_Cap = Tawakoni_Cap = None
if enforce_daily_caps:
    Wylie_Cap    = st.sidebar.number_input("Wylie Plant Capacity (MGD)",    value=DEFAULT_WYLIE_CAP, help="Daily max treated at Wylie")
    Leonard_Cap  = st.sidebar.number_input("Leonard Plant Capacity (MGD)",  value=DEFAULT_LEONARD_CAP, help="Daily max treated at Leonard")
    Tawakoni_Cap = st.sidebar.number_input("Tawakoni Plant Capacity (MGD)", value=DEFAULT_TAWAKONI_CAP,  help="Daily max treated at Tawakoni")

# SCADA upload with reference year selection
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV (columns named by year)", type="csv")
if not uploaded_file:
    st.warning("⚠️ Please upload a SCADA CSV file to continue.")
    st.stop()

scada_df = pd.read_csv(uploaded_file)
if scada_df.empty:
    st.error("❌ Uploaded CSV is empty.")
    st.stop()

cols_as_str = [str(c) for c in scada_df.columns]
ref_year = st.sidebar.selectbox("Reference Year (pick a CSV column)", options=cols_as_str)
SCADA_Data_col = pd.to_numeric(scada_df[ref_year], errors='coerce')
if SCADA_Data_col.isna().any():
    st.error("❌ Selected column contains non-numeric values. Please choose a numeric year column.")
    st.stop()

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
        rW = st.slider("Wylie Ratio", 0.0, 1.0, 0.82, 0.01)
    with col2:
        rL = st.slider("Leonard Ratio", 0.0, 1.0, 0.15, 0.01)
    with col3:
        rT = st.slider("Tawakoni Ratio", 0.0, 1.0, 0.03, 0.01)
    ratio_sum = rW + rL + rT
    if abs(ratio_sum - 1.0) > 1e-6:
        st.error(f"❌ Ratios must sum to 1. Current sum = {ratio_sum:.2f}")
        st.stop()
    st.info(f"Selected ratios → Wylie={rW:.3f}, Leonard={rL:.3f}, Tawakoni={rT:.3f}")

elif mode == "Capacity-based":
    st.subheader("Capacity-based Ratios")
    c1, c2, c3 = st.columns(3)
    with c1:
        capW = st.number_input("Wylie capacity (MGD)", value=DEFAULT_WYLIE_CAP)
    with c2:
        capL = st.number_input("Leonard capacity (MGD)", value=DEFAULT_LEONARD_CAP)
    with c3:
        capT = st.number_input("Tawakoni capacity (MGD)", value=DEFAULT_TAWAKONI_CAP)
    total_cap_ratio = capW + capL + capT
    if total_cap_ratio <= 0:
        st.error("❌ Sum of capacities must be > 0 for capacity-based ratios.")
        st.stop()
    rW, rL, rT = capW/total_cap_ratio, capL/total_cap_ratio, capT/total_cap_ratio
    st.info(f"Selected ratios (capacity-based) → Wylie={rW:.3f}, Leonard={rL:.3f}, Tawakoni={rT:.3f}")

else:
    st.subheader("Optimized Ratios (maximize Texoma usage, respect caps, keep all plants on)")
    rW, rL, rT = optimize_ratios_with_desal(
        Year, SCADA_Data_col, Peak_Day_Demand,
        Mix_Ratio_To_Wylie, Mix_Ratio_To_Leonard,
        Pipe_Cap_To_Wylie, Pipe_Cap_To_Leonard,
        Max_Avg_From_Bois_DARC,
        interval_start_str, interval_end_str, shift_where, include_end,
        desal_policy, floor_wylie, floor_taw, desal_efficiency,
        enforce_daily_caps, Wylie_Cap, Leonard_Cap, Tawakoni_Cap,
        Wylie_floor_opt, Leonard_floor_opt, Tawakoni_floor_opt,
    )
    st.success(f"Optimized ratios → Wylie={rW:.3f}, Leonard={rL:.3f}, Tawakoni={rT:.3f}")

# Show the currently selected ratios under the selector as well
st.caption(f"Current ratios: Wylie={rW:.3f}, Leonard={rL:.3f}, Tawakoni={rT:.3f}")

# -------------------------------
# NEW: Peaks KPIs (available immediately after ratios are set)
# -------------------------------
# Reference peak from selected SCADA column
ref_peak = float(np.max(SCADA_Data_col))
# Plant peaks implied by current ratios
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

# Build a simple signature of the current inputs
_cur_sig = (
    Year, float(Peak_Day_Demand), float(Mix_Ratio_To_Wylie), float(Mix_Ratio_To_Leonard),
    float(Pipe_Cap_To_Wylie), float(Pipe_Cap_To_Leonard), float(Max_Avg_From_Bois_DARC),
    interval_start_str, interval_end_str, include_end, shift_where,
    desal_policy, float(desal_efficiency), float(floor_wylie), float(floor_taw),
    bool(enforce_daily_caps), float(Wylie_Cap or 0), float(Leonard_Cap or 0), float(Tawakoni_Cap or 0),
    float(rW), float(rL), float(rT), str(ref_year), len(SCADA_Data_col), float(SCADA_Data_col.sum()),
    float(Wylie_floor_opt), float(Leonard_floor_opt), float(Tawakoni_floor_opt)
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
        enforce_daily_caps=enforce_daily_caps,
        Wylie_Cap=Wylie_Cap, Leonard_Cap=Leonard_Cap, Tawakoni_Cap=Tawakoni_Cap,
    )

    # Build DataFrame (rounded) and stash in session state
    df = pd.DataFrame({
        "DATE": dates,
        "REFERENCE DAILY DEMAND (MGD)": SCADA,
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

    st.session_state['results'] = {
        "df": df,
        "dates": dates,
        "SCADA": SCADA,
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
        # Also keep peaks for consistency (from UI calcs)
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

    # KPIs (averages as requested)
    k1, k2, k3 , k4 = st.columns(4) 
    with k1:
        st.metric("Annual Avg Wylie WT (MGD)", f"{np.mean(res['Wylie_D']):.1f}")
    with k2:
        st.metric("Annual Avg Leonard WT (MGD)", f"{np.mean(res['Leonard_D']):.1f}")
    with k3:
        st.metric("Annual Avg Tawakoni WT (MGD)", f"{np.mean(res['Tawakoni_D']):.1f}")    
    with k4:
        st.metric("Annual Avg Total Demand (MGD)", f"{np.mean(res['Total_Demand']):.1f}")    
     
#    with k1:
#        st.metric("Achieved Bois d'Arc Avg (MGD)", f"{res['achieved_bois_avg']:.1f}",
#                  delta=f"Target {res['Max_Avg_From_Bois_DARC']:.1f}")
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


    # Cap curtailment reporting
    if res["cap_report"] is not None:
        cr = res["cap_report"]
        msgs = []
        for plant in ["Wylie", "Leonard", "Tawakoni"]:
            if cr[plant]["days"] > 0:
                msgs.append(f"- {plant}: {cr[plant]['days']} day(s) over cap; curtailed **{cr[plant]['curtailed_MG']:.1f} MG** total.")
        if msgs:
            summary = "**Daily Plant Caps:** Flow exceeded caps on:\n" + "\n".join(msgs)
            st.warning(summary)
        else:
            st.success("Daily Plant Caps: No exceedances — all days within caps.")

    # Results table first (scrollable)
    st.subheader("Results (Daily)")
    st.dataframe(res['df'], use_container_width=True, height=420)

    # CSV download (optional)
    with st.expander("Download results as CSV"):
        file_name = st.text_input("CSV file name", value="WaterResults_Desal.csv")
        st.download_button("📥 Download CSV", res['df'].to_csv(index=False), file_name=file_name, mime="text/csv")

    # Plot selections (no recompute)
    st.subheader("Plots")
    plot_options = st.multiselect(
        "Select plots to display:",
        [
            "Texoma Only",
            "Total Demand",
            "Wylie Demand",
            "Wylie Splits (Texoma vs Lavon)",
            "Leonard Demand",
            "Leonard Splits (Texoma vs BoisD)",
            "Bois D'Arc Flow (with limit line)",
            "Tawakoni Demand",
            "Desalinated Water",
        ],
        default=["Texoma Only", "Total Demand", "Bois D'Arc Flow (with limit line)", "Desalinated Water"],
    )

    # Render selected plots
    if "Texoma Only" in plot_options:
        st.pyplot(_plot_series(res['dates'], res['Total_From_Tex'],
                               f"DAILY WATER FROM TEXOMA ({Year_res})",
                               "RAW WATER (MGD)", "blue"))
    if "Total Demand" in plot_options:
        st.pyplot(_plot_series(res['dates'], res['Total_Demand'],
                               f"DAILY TOTAL DEMAND ({Year_res})",
                               "TOTAL WATER DEMAND (MGD)", "purple"))
    if "Wylie Demand" in plot_options:
        st.pyplot(_plot_series(res['dates'], res['Wylie_D'],
                               f"WYLIE DEMAND — {Year_res}",
                               "WYLIE DEMAND (MGD)", "orange"))
    if "Wylie Splits (Texoma vs Lavon)" in plot_options:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(res['dates'], res['Texoma_W'], label="Texoma → Wylie", color="blue")
        ax.plot(res['dates'], res['Lavon_W'],  label="Lavon → Wylie",  color="gray")
        ax.set_title(f"WYLIE SPLITS ({Year_res})"); ax.set_xlabel("DATE"); ax.set_ylabel("MGD")
        ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.35); fig.tight_layout(); st.pyplot(fig)
    if "Leonard Demand" in plot_options:
        st.pyplot(_plot_series(res['dates'], res['Leonard_D'],
                               f"LEONARD DEMAND — {Year_res}",
                               "LEONARD DEMAND (MGD)", "green"))
    if "Leonard Splits (Texoma vs BoisD)" in plot_options:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(res['dates'], res['Texoma_L'], label="Texoma → Leonard", color="blue")
        ax.plot(res['dates'], res['BoisD_L'],  label="Bois d'Arc → Leonard", color="brown")
        ax.set_title(f"LEONARD SPLITS ({Year_res})"); ax.set_xlabel("DATE"); ax.set_ylabel("MGD")
        ax.legend();ax.grid(True, which="both", linestyle="--", alpha=0.35); fig.tight_layout(); st.pyplot(fig)
    if "Bois D'Arc Flow (with limit line)" in plot_options:
        st.pyplot(_plot_series(res['dates'], res['BoisD_L'],
                               f"BOIS D'ARC → LEONARD — {Year_res}",
                               "BOIS D'ARC FLOW (MGD)", "brown", extra_line=res['Max_Avg_From_Bois_DARC']))
    if "Tawakoni Demand" in plot_options:
        st.pyplot(_plot_series(res['dates'], res['Tawakoni_D'],
                               f"TAWAKONI DEMAND — {Year_res}",
                               "TAWAKONI DEMAND (MGD)", "teal"))
    if "Desalinated Water" in plot_options:
        st.pyplot(_plot_series(res['dates'], res['Desal'],
                               f"DESALINATED WATER — {Year_res}",
                               "DESALED FLOW (MGD)", "black"))
else:
    st.info("Set parameters and click **Run Analysis** to compute results. After that, you can switch plots freely without recomputing.")
