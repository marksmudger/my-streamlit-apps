#!/usr/bin/env python3
"""
Investment Allocation & Retirement Savings Simulator — Streamlit App
=====================================================================

Monte Carlo retirement planning tool with interactive visualizations.

Usage
-----
    $ streamlit run "Retirement Investments Model.py"

Dependencies
------------
numpy      (1.19+)  – vectorized computation and random number generation
pandas     (1.2+)   – data manipulation and tabular display
scipy      (1.6+)   – differential_evolution optimizer
streamlit  (1.28+)  – web application framework
plotly     (5.0+)   – interactive charts

What the tool does
------------------
1. Collects basic financial inputs from an interactive sidebar form.
2. Simulates the accumulation phase with AR(1) income and a linear glidepath.
3. Simulates the withdrawal phase to capture sequence-of-returns risk.
4. Optimizes the starting portfolio allocation to maximize withdrawal survival.
5. Renders interactive charts covering trajectories, allocation, survival,
   scenarios, and return-assumption sensitivity.

Known model limitations
-----------------------
- The glidepath is linear and deterministic (not dynamically optimized).
- Income autocorrelation is fixed at 0.3 (Guvenen 2009 literature average).
- Social Security projections assume 2% annual COLA; actual adjustments vary.
- Past return and volatility data (1926-2023) may not repeat in the future.
- Tax treatment is not modeled; all figures are pre-tax.

Author:       Mark Smith
Version:      4.0.0
Last Updated: February 2026
"""

# ── Standard-library imports ──────────────────────────────────────────────────
import math
from dataclasses import dataclass

# ── Third-party imports ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retirement Investment Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module-level RNG ──────────────────────────────────────────────────────────
rng = np.random.default_rng()


# =============================================================================
# Configuration Constants
# =============================================================================

SIMULATION_COUNT  = 5_000   # paths for final reporting (reduced for web responsiveness)
OPTIMIZATION_SIMS =   500   # paths per objective call during optimisation

DEFAULT_TARGET_RATE = 75.0  # primary target probability (%)
MIN_SUCCESS_RATE    = 60.0  # below this → strong caution
HIGH_SUCCESS_RATE   = 85.0  # above this → congratulatory

MAX_SINGLE_ASSET         = 0.80
ALTERNATIVES_MAX_SHARE   = 0.10
ALTERNATIVES_HARD_PENALTY = 1e12

HISTORIC_NOMINAL_RETURN = 0.06   # 6% long-run nominal portfolio return
HISTORIC_INFLATION      = 0.02   # 2% Fed inflation target

# AR(1) income autocorrelation (Guvenen 2009 literature average)
INCOME_AUTOCORR = 0.30


# =============================================================================
# Asset-Class Data
# =============================================================================
# Nominal (pre-inflation) annual return statistics from 1926-2023.
# Sources: S&P 500, Bloomberg US Corporate IG, 10-yr Treasury,
#          NAREIT All REITs, S&P GSCI, 3-month T-Bill.

ASSET_CLASSES = {
    "Stocks":             {"expected_return": 0.102, "stdev": 0.158},
    "Corporate Bonds":    {"expected_return": 0.059, "stdev": 0.072},
    "Government Bonds":   {"expected_return": 0.050, "stdev": 0.057},
    "Real Estate (REIT)": {"expected_return": 0.098, "stdev": 0.174},
    "Commodities":        {"expected_return": 0.047, "stdev": 0.153},
    "Cash":               {"expected_return": 0.033, "stdev": 0.008},
}

ASSET_NAMES = list(ASSET_CLASSES.keys())
NUM_ASSETS  = len(ASSET_NAMES)

ASSET_MEANS = np.array([ASSET_CLASSES[a]["expected_return"] for a in ASSET_NAMES])
ASSET_STDS  = np.array([ASSET_CLASSES[a]["stdev"]           for a in ASSET_NAMES])

# Asset palette — consistent across all charts
ASSET_COLORS = {
    "Stocks":             "#1565C0",
    "Corporate Bonds":    "#2E7D32",
    "Government Bonds":   "#558B2F",
    "Real Estate (REIT)": "#E65100",
    "Commodities":        "#AD1457",
    "Cash":               "#546E7A",
}

# Annual correlation matrix (row/col order = ASSET_NAMES)
# Conservative long-run estimates (Ilmanen "Expected Returns";
# Brinson, Hood & Beebower).
#   0 Stocks  1 Corp Bonds  2 Govt Bonds  3 REIT  4 Commodities  5 Cash
CORRELATION_MATRIX = np.array(
    [
        [ 1.00, -0.10, -0.20,  0.60,  0.15,  0.00],  # Stocks
        [-0.10,  1.00,  0.80, -0.05,  0.05,  0.10],  # Corporate Bonds
        [-0.20,  0.80,  1.00, -0.10, -0.05,  0.20],  # Government Bonds
        [ 0.60, -0.05, -0.10,  1.00,  0.10,  0.00],  # REIT
        [ 0.15,  0.05, -0.05,  0.10,  1.00,  0.00],  # Commodities
        [ 0.00,  0.10,  0.20,  0.00,  0.00,  1.00],  # Cash
    ],
    dtype=float,
)

_sigma_ln = np.log1p(ASSET_STDS)
ASSET_COV  = np.outer(_sigma_ln, _sigma_ln) * CORRELATION_MATRIX

# Ensure positive semi-definiteness (required for multivariate_normal)
_min_eig = np.linalg.eigvalsh(ASSET_COV).min()
if _min_eig < 0:
    ASSET_COV -= _min_eig * np.eye(NUM_ASSETS)

MU_LN = np.log1p(ASSET_MEANS) - 0.5 * _sigma_ln ** 2

# Conservative retirement-phase weights (Vanguard Target Retirement Income
# profile): 30% stocks / 20% corp bonds / 30% govt bonds / 20% cash
CONSERVATIVE_RETIREMENT_WEIGHTS = np.array([0.30, 0.20, 0.30, 0.00, 0.00, 0.20])


# =============================================================================
# Simulation Configuration
# =============================================================================

@dataclass
class SimConfig:
    """Immutable bundle of user inputs and derived parameters passed to
    all simulation functions instead of relying on module-level globals."""
    current_age:         int
    retirement_age:      int
    years_to_simulate:   int
    retirement_years:    int
    current_income:      float
    income_growth_rate:  float
    income_growth_range: float
    current_savings:     float
    savings_rate:        float
    other_savings:       float
    ss_annual_benefit:   float
    replacement_rate:    float
    annuity_factor:      float
    min_bond:            float


# =============================================================================
# Utility / Helper Functions
# =============================================================================

def format_currency(value: float) -> str:
    """Compact dollar string: '$2.5M' for millions, '$950,000' otherwise."""
    return f"${value / 1_000_000:,.1f}M" if value >= 1_000_000 else f"${value:,.0f}"


def _retirement_params(ret_age: int):
    """
    Return (retirement_years, replacement_rate, real_return) based on
    SSA actuarial tables and standard planning guidelines.

    Life expectancy (simplified SSA tables):
      <65  → 85  |  65+ → 87  |  75+ → 90

    Replacement rate (fraction of final salary needed):
      <62  → 80%  |  62+ → 75%  |  67+ → 70%
    """
    life_expectancy = 85 if ret_age < 65 else (90 if ret_age >= 75 else 87)
    ret_years       = life_expectancy - ret_age
    repl_rate       = 0.80 if ret_age < 62 else (0.70 if ret_age >= 67 else 0.75)
    real_return     = (1 + HISTORIC_NOMINAL_RETURN) / (1 + HISTORIC_INFLATION) - 1
    return ret_years, repl_rate, real_return


def _min_bond_alloc(years_to_retirement: int) -> float:
    """
    Minimum combined bond + cash allocation for the accumulation horizon.
    Implements a simple glidepath rule enforced inside the optimiser.
      >20 yr → 5%  |  ≤20 yr → 10%  |  ≤10 yr → 20%  |  ≤5 yr → 30%
    """
    if years_to_retirement <= 5:
        return 0.30
    elif years_to_retirement <= 10:
        return 0.20
    elif years_to_retirement <= 20:
        return 0.10
    return 0.05


def build_config(
    current_age, retirement_age, current_income, income_growth_rate,
    income_growth_range, current_savings, savings_rate, other_savings,
    ss_annual_benefit,
) -> SimConfig:
    """Construct a SimConfig from raw user inputs, computing all derived fields."""
    years_to_simulate = int(math.ceil(retirement_age - current_age))
    ret_years, repl_rate, real_return = _retirement_params(retirement_age)
    r = real_return
    n = ret_years
    annuity_factor = (1.0 - (1.0 + r) ** (-n)) / r if r != 0 else float(n)
    return SimConfig(
        current_age=int(current_age),
        retirement_age=int(retirement_age),
        years_to_simulate=years_to_simulate,
        retirement_years=ret_years,
        current_income=float(current_income),
        income_growth_rate=float(income_growth_rate),
        income_growth_range=float(income_growth_range),
        current_savings=float(current_savings),
        savings_rate=float(savings_rate),
        other_savings=float(other_savings),
        ss_annual_benefit=float(ss_annual_benefit),
        replacement_rate=repl_rate,
        annuity_factor=annuity_factor,
        min_bond=_min_bond_alloc(years_to_simulate),
    )


def estimate_ss_benefit(
    current_income: float,
    current_age: int,
    retirement_age: int,
) -> float:
    """
    Rough estimate of annual Social Security benefit at the given retirement age,
    expressed in today's dollars.

    Method:
    1. Approximate AIME (Average Indexed Monthly Earnings) as 75% of current
       monthly income — a proxy for career-average indexed earnings.
    2. Apply the 2024 SSA bend-point formula to compute the PIA
       (Primary Insurance Amount — the benefit at Full Retirement Age).
    3. Adjust for early or delayed claiming relative to Full Retirement Age.

    This is a ballpark figure only. Actual benefits depend on the user's
    complete earnings history. ssa.gov/myaccount provides a personalized estimate.
    """
    if current_income <= 0:
        return 0.0

    # ── Full Retirement Age (FRA) from approximate birth year ─────────────────
    birth_year = 2026 - current_age
    if birth_year >= 1960:
        fra = 67.0
    elif birth_year >= 1943:
        fra = 66.0
    else:
        fra = 65.0

    # ── Estimate AIME ─────────────────────────────────────────────────────────
    # 75% of current monthly income roughly approximates career-average indexed
    # earnings for a mid-career worker. Cap at the 2024 taxable earnings maximum.
    taxable_max_monthly = 168_600 / 12   # 2024 Social Security taxable wage cap
    aime = min(current_income / 12, taxable_max_monthly) * 0.75

    # ── PIA via 2024 SSA bend points ──────────────────────────────────────────
    bp1, bp2 = 1_174.0, 7_078.0          # monthly dollar thresholds
    if aime <= bp1:
        pia_monthly = 0.90 * aime
    elif aime <= bp2:
        pia_monthly = 0.90 * bp1 + 0.32 * (aime - bp1)
    else:
        pia_monthly = 0.90 * bp1 + 0.32 * (bp2 - bp1) + 0.15 * (aime - bp2)

    # ── Adjust for claiming age vs FRA ────────────────────────────────────────
    years_diff = retirement_age - fra
    if years_diff >= 0:
        # Delayed claiming: +8% per year beyond FRA (credits stop at age 70)
        adjustment = 1.0 + 0.08 * min(years_diff, 70 - fra)
    else:
        # Early claiming: −5/9% per month for first 36 months early,
        # then −5/12% per month beyond that
        months_early = abs(years_diff) * 12
        if months_early <= 36:
            reduction = months_early * (5.0 / 9.0) / 100.0
        else:
            reduction = (36 * (5.0 / 9.0) + (months_early - 36) * (5.0 / 12.0)) / 100.0
        adjustment = max(0.70, 1.0 - reduction)   # floor at 70% of PIA

    annual = pia_monthly * adjustment * 12
    return float(round(annual / 500) * 500)   # round to nearest $500


def success_color(rate: float) -> str:
    if rate >= HIGH_SUCCESS_RATE:
        return "#2E7D32"
    elif rate >= DEFAULT_TARGET_RATE:
        return "#388E3C"
    elif rate >= MIN_SUCCESS_RATE:
        return "#F57C00"
    return "#C62828"


def success_label(rate: float) -> str:
    if rate >= HIGH_SUCCESS_RATE:
        return "Excellent"
    elif rate >= DEFAULT_TARGET_RATE:
        return "On Track"
    elif rate >= MIN_SUCCESS_RATE:
        return "Caution"
    return "At Risk"


# =============================================================================
# Simulation Engine  (vectorized Monte Carlo)
# =============================================================================

def run_simulations(
    weights: np.ndarray,
    cfg: SimConfig,
    n_sims: int,
    years: int = None,
    sr: float = None,
    mu_override: np.ndarray = None,
    return_paths: bool = False,
):
    """
    Batch Monte Carlo simulation of the accumulation phase.

    Model features:
    • AR(1) income growth with autocorrelation INCOME_AUTOCORR = 0.30.
    • Linear glidepath blending from growth weights to
      CONSERVATIVE_RETIREMENT_WEIGHTS over the accumulation horizon.
    • Correlated multi-asset log-normal returns via Cholesky decomposition.
    • SS nominal projection via 2% COLA.

    Parameters
    ----------
    weights       : (n_assets,) starting (growth) portfolio weights
    cfg           : SimConfig holding all model parameters
    n_sims        : number of independent Monte Carlo paths
    years         : override accumulation horizon (default cfg.years_to_simulate)
    sr            : savings-rate override for scenario analysis
    mu_override   : substitute log-return mean vector (sensitivity analysis)
    return_paths  : if True, also return the (n_sims, years+1) path matrix

    Returns
    -------
    savings, final_income, annual_withdrawal, goal_met_proxy
    — and paths (if return_paths=True)
    """
    weights = np.asarray(weights, dtype=float)
    _years = years if years is not None else cfg.years_to_simulate
    _sr    = sr    if sr    is not None else cfg.savings_rate
    _mu    = mu_override if mu_override is not None else MU_LN

    # ── AR(1) income paths ────────────────────────────────────────────────────
    sigma_eps = cfg.income_growth_range * np.sqrt(1.0 - INCOME_AUTOCORR ** 2)
    eps = rng.normal(0.0, sigma_eps, (_years, n_sims))
    income_shocks = np.empty((_years, n_sims))
    income_shocks[0] = cfg.income_growth_rate + eps[0]
    for t in range(1, _years):
        income_shocks[t] = (
            cfg.income_growth_rate
            + INCOME_AUTOCORR * (income_shocks[t - 1] - cfg.income_growth_rate)
            + eps[t]
        )
    income_paths  = cfg.current_income * np.cumprod(1.0 + income_shocks, axis=0)
    final_income  = income_paths[-1]                              # (n_sims,)
    contributions = income_paths * _sr + cfg.other_savings        # (_years, n_sims)

    # ── Correlated multi-asset log-normal returns ─────────────────────────────
    raw_log_returns = rng.multivariate_normal(_mu, ASSET_COV, size=(n_sims * _years))
    simple_returns  = np.expm1(raw_log_returns).reshape(n_sims, _years, NUM_ASSETS)

    # ── Accumulate with glidepath ─────────────────────────────────────────────
    savings = np.full(n_sims, float(cfg.current_savings))
    if return_paths:
        paths = np.zeros((n_sims, _years + 1))
        paths[:, 0] = savings

    for t in range(_years):
        alpha = (t / (_years - 1)) if _years > 1 else 0.0
        w_t   = (1.0 - alpha) * weights + alpha * CONSERVATIVE_RETIREMENT_WEIGHTS
        w_t   = w_t / w_t.sum()
        r_t   = simple_returns[:, t, :] @ w_t               # (n_sims,)
        savings = savings * (1.0 + r_t) + contributions[t]
        if return_paths:
            paths[:, t + 1] = savings

    # ── SS nominal projection & required withdrawal ───────────────────────────
    ss_nominal        = cfg.ss_annual_benefit * (1.0 + HISTORIC_INFLATION) ** _years
    annual_withdrawal = np.maximum(0.0, cfg.replacement_rate * final_income - ss_nominal)
    goal_met_proxy    = savings >= annual_withdrawal * cfg.annuity_factor

    if return_paths:
        return savings, final_income, annual_withdrawal, goal_met_proxy, paths
    return savings, final_income, annual_withdrawal, goal_met_proxy


# =============================================================================
# Withdrawal-Phase Simulation
# =============================================================================

def run_withdrawal_simulation(
    nest_egg: np.ndarray,
    annual_withdrawal: np.ndarray,
    withdrawal_weights: np.ndarray,
    n_years: int,
    return_survival_curve: bool = False,
):
    """
    Simulate the retirement drawdown phase to capture sequence-of-returns risk.

    portfolio[t] = portfolio[t-1] × (1 + r[t]) − withdrawal × (1+inflation)^t

    A path "survives" if the portfolio stays above zero for the full horizon.

    Parameters
    ----------
    nest_egg              : (n_sims,) portfolio value at retirement date
    annual_withdrawal     : (n_sims,) first-year draw needed from portfolio
    withdrawal_weights    : (n_assets,) conservative retirement allocation
    n_years               : retirement horizon to simulate
    return_survival_curve : if True, also return year-by-year survival rates

    Returns
    -------
    survived : (n_sims,) bool
    survival_curve : (n_years,) float, fraction surviving each year [optional]
    """
    n_sims  = len(nest_egg)
    savings = nest_egg.astype(float).copy()

    raw       = rng.multivariate_normal(MU_LN, ASSET_COV, size=(n_sims * n_years))
    port_ret  = np.expm1(raw).reshape(n_sims, n_years, NUM_ASSETS) @ withdrawal_weights

    survival_curve = np.zeros(n_years) if return_survival_curve else None

    for t in range(n_years):
        savings = savings * (1.0 + port_ret[:, t])
        savings -= annual_withdrawal * (1.0 + HISTORIC_INFLATION) ** t
        savings = np.maximum(savings, 0.0)
        if return_survival_curve:
            survival_curve[t] = (savings > 0).mean()

    survived = savings > 0
    return (survived, survival_curve) if return_survival_curve else survived


# =============================================================================
# Portfolio Optimiser
# =============================================================================

def optimize_portfolio(cfg: SimConfig) -> np.ndarray:
    """
    Differential Evolution optimizer that maximizes withdrawal-survival success
    rate (uses the faster proxy objective during search).

    Objective: −success_rate − median_savings / 1e9
    Hard constraints:
      • REIT + Commodities ≤ ALTERNATIVES_MAX_SHARE (10%)
      • Bonds + Cash ≥ cfg.min_bond  (glidepath floor)
    """
    alt_idx  = [ASSET_NAMES.index(a) for a in ("Real Estate (REIT)", "Commodities")]
    bond_idx = [ASSET_NAMES.index(a) for a in ("Corporate Bonds", "Government Bonds", "Cash")]

    def objective(x: np.ndarray) -> float:
        w = np.clip(x, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(x) / x.size
        if w[alt_idx].sum() > ALTERNATIVES_MAX_SHARE:
            return ALTERNATIVES_HARD_PENALTY
        if w[bond_idx].sum() < cfg.min_bond:
            return ALTERNATIVES_HARD_PENALTY
        savings, _, _, goal_met = run_simulations(w, cfg, n_sims=OPTIMIZATION_SIMS)
        return -goal_met.mean() - float(np.median(savings)) / 1e9

    result = differential_evolution(
        objective,
        bounds=[(0.0, 1.0)] * NUM_ASSETS,
        maxiter=40,
        popsize=15,
        disp=False,
        seed=None,
    )
    w = np.clip(result.x, 0.0, None)
    return w / w.sum() if w.sum() > 0 else np.ones(NUM_ASSETS) / NUM_ASSETS


# =============================================================================
# Scenario / Sensitivity Helpers
# =============================================================================

def run_scenario(
    cfg: SimConfig,
    weights: np.ndarray,
    sr_override: float = None,
    extra_years: int = 0,
    n_sims: int = 2_000,
) -> float:
    """Return withdrawal-survival success rate (%) under modified inputs."""
    _years = cfg.years_to_simulate + extra_years
    savings, _, wd, _ = run_simulations(weights, cfg, n_sims=n_sims,
                                        years=_years, sr=sr_override)
    survived = run_withdrawal_simulation(
        savings, wd, CONSERVATIVE_RETIREMENT_WEIGHTS, cfg.retirement_years
    )
    return float(survived.mean() * 100)


def run_sensitivity(cfg: SimConfig, weights: np.ndarray, delta: float,
                    n_sims: int = 2_000) -> float:
    """Return withdrawal-survival success rate (%) with mu shifted by delta."""
    savings, _, wd, _ = run_simulations(weights, cfg, n_sims=n_sims,
                                        mu_override=MU_LN + delta)
    survived = run_withdrawal_simulation(
        savings, wd, CONSERVATIVE_RETIREMENT_WEIGHTS, cfg.retirement_years
    )
    return float(survived.mean() * 100)


# =============================================================================
# Visualisation Functions
# =============================================================================

_TEMPLATE = "plotly_white"
_FONT     = dict(family="Inter, Arial, sans-serif")


def fig_savings_fan(paths: np.ndarray, cfg: SimConfig) -> go.Figure:
    """
    Fan chart: percentile bands of simulated portfolio values over time.

    The corridor widens as uncertainty compounds. Bands shown: P5–P95,
    P10–P90, P25–P75, and the median (P50).
    """
    ages  = np.arange(cfg.current_age, cfg.retirement_age + 1)
    pcts  = [5, 10, 25, 50, 75, 90, 95]
    bands = np.percentile(paths, pcts, axis=0)   # (7, n_years+1)

    # Band fill pairs: (upper_idx, lower_idx, opacity, legend)
    fill_layers = [
        (6, 0, "rgba(21,101,192,0.06)", "5th to 95th Percentile"),
        (5, 1, "rgba(21,101,192,0.12)", "10th to 90th Percentile"),
        (4, 2, "rgba(21,101,192,0.22)", "25th to 75th Percentile"),
    ]

    fig = go.Figure()

    for upper_i, lower_i, color, name in fill_layers:
        x_fill = np.concatenate([ages, ages[::-1]])
        y_fill = np.concatenate([bands[upper_i], bands[lower_i][::-1]])
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill,
            fill="toself", fillcolor=color,
            line=dict(width=0),
            name=name, showlegend=True,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=ages, y=bands[3],
        line=dict(color="#1565C0", width=2.5),
        name="Median (50th Percentile)",
        hovertemplate="Age %{x}<br>Median: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        template=_TEMPLATE, font=_FONT,
        xaxis=dict(title="Age", dtick=5, tickmode="linear"),
        yaxis=dict(title="Portfolio Value", tickformat="$,.0f"),
        legend=dict(orientation="h", y=-0.18, x=0),
        hovermode="x unified",
        height=430,
        margin=dict(t=20, b=60),
    )
    return fig


def fig_portfolio_donut(weights: np.ndarray, current_savings: float) -> go.Figure:
    """
    Donut chart of the optimised portfolio allocation.
    Hover shows weight, dollar amount, expected return, and volatility.
    """
    colors = [ASSET_COLORS[a] for a in ASSET_NAMES]

    hover_texts = [
        (
            f"<b>{name}</b><br>"
            f"Allocation: {w * 100:.1f}%<br>"
            f"Amount: ${w * current_savings:,.0f}<br>"
            f"Exp. Return: {ASSET_CLASSES[name]['expected_return'] * 100:.1f}%<br>"
            f"Annual Volatility: {ASSET_CLASSES[name]['stdev'] * 100:.1f}%"
            "<extra></extra>"
        )
        for name, w in zip(ASSET_NAMES, weights)
    ]

    fig = go.Figure(go.Pie(
        labels=ASSET_NAMES,
        values=weights * 100,
        hole=0.52,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        hovertemplate=hover_texts,
        textinfo="label+percent",
        textfont=dict(size=11),
        sort=False,
    ))

    center_label = (
        f"${current_savings / 1e3:,.0f}K" if current_savings < 1e6
        else f"${current_savings / 1e6:,.1f}M"
    )

    fig.update_layout(
        template=_TEMPLATE, font=_FONT,
        annotations=[dict(
            text=f"<b>{center_label}</b><br>Today",
            x=0.5, y=0.5, font_size=13, showarrow=False,
        )],
        legend=dict(orientation="v", x=1.02),
        height=400,
        margin=dict(t=20),
    )
    return fig


def fig_glidepath(weights: np.ndarray, cfg: SimConfig) -> go.Figure:
    """
    Stacked area chart showing how the portfolio de-risks linearly over time,
    blending from the optimised growth weights to the conservative retirement
    allocation — identical in principle to a Target-Date Fund.
    """
    n = cfg.years_to_simulate
    all_weights = []
    for t in range(n + 1):
        alpha = min(t / (n - 1), 1.0) if n > 1 else 0.0
        w = (1 - alpha) * weights + alpha * CONSERVATIVE_RETIREMENT_WEIGHTS
        all_weights.append(w / w.sum())

    all_weights = np.array(all_weights)   # (n+1, n_assets)
    ages = np.arange(cfg.current_age, cfg.retirement_age + 1)

    fig = go.Figure()
    for i, name in enumerate(ASSET_NAMES):
        fig.add_trace(go.Scatter(
            x=ages, y=all_weights[:, i] * 100,
            name=name, mode="lines",
            stackgroup="one",
            fillcolor=ASSET_COLORS[name],
            line=dict(width=0.5, color=ASSET_COLORS[name]),
            hovertemplate=f"<b>{name}</b><br>Age %{{x}}<br>%{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        template=_TEMPLATE, font=_FONT,
        xaxis=dict(title="Age", dtick=5, tickmode="linear"),
        yaxis=dict(title="Allocation (%)", range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", y=-0.22, x=0),
        hovermode="x unified",
        height=380,
        margin=dict(t=20, b=70),
    )
    return fig


def fig_withdrawal_survival(survival_curve: np.ndarray, cfg: SimConfig) -> go.Figure:
    """
    Area chart of portfolio survival probability through each retirement year.

    Horizontal colour zones mark the four outcome bands (Excellent / On Track /
    Caution / At Risk) so performance can be read at a glance.
    """
    ages = np.arange(cfg.retirement_age + 1, cfg.retirement_age + cfg.retirement_years + 1)

    fig = go.Figure()

    # Outcome zones
    zones = [
        (HIGH_SUCCESS_RATE, 100,               "rgba(46,125,50,0.07)",  "Excellent"),
        (DEFAULT_TARGET_RATE, HIGH_SUCCESS_RATE, "rgba(255,193,7,0.09)", "On Track"),
        (MIN_SUCCESS_RATE, DEFAULT_TARGET_RATE,  "rgba(255,152,0,0.11)", "Caution"),
        (0,               MIN_SUCCESS_RATE,      "rgba(198,40,40,0.09)", "At Risk"),
    ]
    for y0, y1, color, label in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=11)

    fig.add_trace(go.Scatter(
        x=ages, y=survival_curve * 100,
        mode="lines",
        line=dict(color="#1565C0", width=3),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.10)",
        hovertemplate="Age %{x}<br>Survival: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))

    fig.add_hline(
        y=DEFAULT_TARGET_RATE, line_dash="dash", line_color="#9E9E9E",
        annotation_text=f"{DEFAULT_TARGET_RATE:.0f}% target",
        annotation_position="right",
    )

    fig.update_layout(
        template=_TEMPLATE, font=_FONT,
        xaxis=dict(title="Age in Retirement", dtick=5, tickmode="linear"),
        yaxis=dict(title="Paths Still Solvent (%)",
                   range=[0, 105], ticksuffix="%"),
        height=400,
        margin=dict(t=20, r=120),
    )
    return fig


def fig_scenarios(base_rate: float, scenarios: dict) -> go.Figure:
    """
    Horizontal bar chart comparing success rates across what-if scenarios.
    Bar colour encodes outcome category (green → red).
    """
    labels = ["Current plan"] + list(scenarios.keys())
    rates  = [base_rate]      + list(scenarios.values())
    colors = [success_color(r) for r in rates]

    fig = go.Figure(go.Bar(
        x=rates, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Success Rate: %{x:.1f}%<extra></extra>",
        cliponaxis=False,
    ))

    fig.add_vline(
        x=DEFAULT_TARGET_RATE, line_dash="dash", line_color="#9E9E9E",
        annotation_text=f"{DEFAULT_TARGET_RATE:.0f}% target",
        annotation_position="top",
    )

    n_bars = len(labels)
    fig.update_layout(
        template=_TEMPLATE, font=_FONT,
        xaxis=dict(title="Withdrawal Survival Rate (%)",
                   range=[0, 110], ticksuffix="%"),
        yaxis=dict(title="", autorange="reversed"),
        height=80 + n_bars * 46,
        margin=dict(l=240, r=60, t=20, b=40),
    )
    return fig


def fig_sensitivity(base_rate: float, minus1: float, minus2: float) -> go.Figure:
    """
    Bar chart showing how success rate degrades if future returns are
    structurally lower than the 1926-2023 historical average.
    """
    labels = ["Baseline (historical average)", "Returns −1% per year", "Returns −2% per year"]
    rates  = [base_rate, minus1, minus2]
    colors = [success_color(r) for r in rates]

    fig = go.Figure(go.Bar(
        x=labels, y=rates,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Success Rate: %{y:.1f}%<extra></extra>",
        cliponaxis=False,
    ))

    fig.add_hline(
        y=DEFAULT_TARGET_RATE, line_dash="dash", line_color="#9E9E9E",
        annotation_text=f"{DEFAULT_TARGET_RATE:.0f}% target",
        annotation_position="right",
    )

    fig.update_layout(
        template=_TEMPLATE, font=_FONT,
        xaxis=dict(title=""),
        yaxis=dict(title="Withdrawal Survival Rate (%)",
                   range=[0, 115], ticksuffix="%"),
        height=380,
        margin=dict(t=40, b=40),
    )
    return fig


_RISK_RETURN_TEXT_POSITIONS = {
    "Stocks":             "top center",
    "Corporate Bonds":    "top right",
    "Government Bonds":   "bottom left",
    "Real Estate (REIT)": "top center",
    "Commodities":        "top left",
    "Cash":               "bottom right",
}


def fig_risk_return() -> go.Figure:
    """
    Scatter chart of expected return vs. annual volatility for each asset class
    in the model universe. Bubble size is proportional to Sharpe ratio
    (excess return / volatility, assuming 3.3% cash rate).
    """
    rf = ASSET_CLASSES["Cash"]["expected_return"]

    fig = go.Figure()
    for name in ASSET_NAMES:
        mu  = ASSET_CLASSES[name]["expected_return"] * 100
        sig = ASSET_CLASSES[name]["stdev"] * 100
        sharpe = (ASSET_CLASSES[name]["expected_return"] - rf) / ASSET_CLASSES[name]["stdev"]
        size = max(12, sharpe * 40)

        fig.add_trace(go.Scatter(
            x=[sig], y=[mu],
            mode="markers+text",
            marker=dict(size=size, color=ASSET_COLORS[name],
                        line=dict(width=2, color="white"), opacity=0.9),
            text=[name],
            textposition=_RISK_RETURN_TEXT_POSITIONS[name],
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Expected Return: {mu:.1f}%<br>"
                f"Annual Volatility: {sig:.1f}%<br>"
                f"Sharpe Ratio: {sharpe:.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=_TEMPLATE, font=_FONT,
        xaxis=dict(title="Annual Volatility (Standard Deviation)", ticksuffix="%"),
        yaxis=dict(title="Expected Annual Return", ticksuffix="%"),
        showlegend=False,
        height=400,
        margin=dict(t=20),
    )
    return fig


def fig_correlation_heatmap() -> go.Figure:
    """
    Annotated heatmap of the asset-return correlation matrix used by the model.
    """
    fig = go.Figure(go.Heatmap(
        z=CORRELATION_MATRIX,
        x=ASSET_NAMES,
        y=ASSET_NAMES,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in CORRELATION_MATRIX],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate=(
            "<b>%{y} × %{x}</b><br>Correlation: %{z:.2f}<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(title="Correlation", tickformat=".1f"),
    ))

    fig.update_layout(
        template=_TEMPLATE, font=_FONT,
        xaxis=dict(side="bottom", tickangle=-30),
        height=380,
        margin=dict(t=20, b=80),
    )
    return fig


def fig_withdrawal_distribution(annual_withdrawal_results: np.ndarray) -> go.Figure:
    """
    Histogram of first-year withdrawal amounts required from the portfolio
    (post Social Security), across all simulation paths.
    """
    vals = annual_withdrawal_results / 1_000   # convert to thousands

    fig = go.Figure(go.Histogram(
        x=vals,
        nbinsx=60,
        marker_color="#1565C0",
        opacity=0.75,
        hovertemplate="$%{x:.0f}K/yr<br>Count: %{y}<extra></extra>",
    ))

    p25 = float(np.percentile(vals, 25))
    p50 = float(np.percentile(vals, 50))
    p75 = float(np.percentile(vals, 75))

    for (pct, label, color), y_pos in zip(
        [
            (p25, "25th Percentile", "#F57C00"),
            (p50, "Median",          "#1565C0"),
            (p75, "75th Percentile", "#C62828"),
        ],
        [0.95, 0.78, 0.61],
    ):
        fig.add_vline(
            x=pct, line_dash="dash", line_color=color,
            annotation_text=f"{label}: ${pct:.0f}K",
            annotation_yref="paper",
            annotation_y=y_pos,
            annotation_yanchor="bottom",
            annotation_xanchor="left",
        )

    fig.update_layout(
        template=_TEMPLATE, font=_FONT,
        xaxis=dict(title="First-Year Portfolio Withdrawal ($K/yr)", tickprefix="$", ticksuffix="K"),
        yaxis=dict(title="Number of Simulation Paths"),
        height=360,
        margin=dict(t=20),
        showlegend=False,
    )
    return fig


# =============================================================================
# Streamlit Application
# =============================================================================

def _sidebar_inputs():
    """Render sidebar form and return raw input values."""
    with st.sidebar:
        st.markdown("## Your Financial Profile")
        st.caption(
            "All dollar amounts are in today's dollars (not adjusted for future inflation). "
            "Results are shown in these same unadjusted, or 'nominal', terms."
        )

        st.markdown("### Personal")
        current_age = st.number_input(
            "Current age",
            min_value=18, max_value=80, value=40, step=1,
            help="Your age today in whole years.",
        )
        retirement_age = st.number_input(
            "Target retirement age",
            min_value=int(current_age) + 1, max_value=90, value=65, step=1,
            help=(
                "The age at which you plan to stop working. Affects the "
                "accumulation horizon, life-expectancy assumption, and income "
                "replacement rate."
            ),
        )

        st.markdown("### Income")
        current_income = st.number_input(
            "Annual gross income ($)",
            min_value=0, max_value=10_000_000, value=100_000, step=5_000,
            help="Your current pre-tax annual income.",
        )
        income_growth_pct = st.slider(
            "Expected annual income growth (%)",
            min_value=0.0, max_value=15.0, value=4.0, step=0.5,
            format="%.1f%%",
            help=(
                "Average annual raise or income growth rate. "
                "Typical range: 2 to 5%. Used as the long-run average in a statistical model "
                "where each year's income growth is partially influenced by the previous year's."
            ),
        )
        income_vol_pct = st.slider(
            "Income growth variability: Standard Deviation (σ) (%)",
            min_value=0.0, max_value=12.0, value=2.0, step=0.5,
            format="%.1f%%",
            help=(
                "Year-to-year standard deviation (σ) of income growth: "
                "how much your annual raises vary from year to year. "
                "Higher values indicate a more unpredictable career trajectory. "
                "A typical office-worker value is 1 to 3%."
            ),
        )

        st.markdown("### Savings")
        current_savings = st.number_input(
            "Current retirement savings ($)",
            min_value=0, max_value=50_000_000, value=50_000, step=5_000,
            help=(
                "Total value of all retirement accounts today, "
                "including 401(k)s, IRAs, Roth IRAs, and taxable brokerage accounts."
            ),
        )
        savings_rate_pct = st.slider(
            "Savings rate (% of income)",
            min_value=0, max_value=80, value=15, step=1,
            format="%d%%",
            help=(
                "Fraction of gross income contributed to retirement savings each year. "
                "A common rule of thumb is 15%; higher is better."
            ),
        )
        other_savings = st.number_input(
            "Additional fixed annual savings ($)",
            min_value=0, max_value=500_000, value=0, step=1_000,
            help=(
                "Fixed-dollar contributions outside your regular income each year, "
                "such as inheritances, rental income, or side income."
            ),
        )

        st.markdown("### Social Security")
        include_ss = st.checkbox(
            "Include Social Security benefit",
            value=True,
            help=(
                "Social Security offsets the nest egg required from your portfolio. "
                "Check ssa.gov/myaccount for your personalized estimate."
            ),
        )
        ss_benefit = 0.0
        if include_ss:
            estimated_ss = estimate_ss_benefit(
                current_income, int(current_age), int(retirement_age)
            )
            # Pre-fill with the current estimate; reset if the estimate changes
            # (e.g. because the user updated their income or retirement age).
            if (
                "ss_benefit_input" not in st.session_state
                or st.session_state.get("ss_benefit_estimate") != int(estimated_ss)
            ):
                st.session_state["ss_benefit_input"] = int(estimated_ss)
            st.session_state["ss_benefit_estimate"] = int(estimated_ss)
            ss_benefit = float(st.number_input(
                "Expected annual Social Security benefit (today's $)",
                min_value=0, max_value=60_000, step=500,
                key="ss_benefit_input",
                help=(
                    "Your estimated annual Social Security income at retirement, "
                    "expressed in today's purchasing power. The model projects it "
                    "forward using a 2% annual cost-of-living adjustment. "
                    "The pre-filled value is a rough estimate based on your "
                    "income and retirement age; edit it if you have a more accurate figure."
                ),
            ))
            st.caption(
                f"Rough estimate based on your income and retirement age: "
                f"**~${estimated_ss:,.0f}/yr**. "
                "For your actual benefit, visit "
                "[ssa.gov/myaccount](https://www.ssa.gov/myaccount/)."
            )

        st.divider()
        run = st.button("Run Analysis", type="primary", use_container_width=True)

    return dict(
        current_age=int(current_age),
        retirement_age=int(retirement_age),
        current_income=float(current_income),
        income_growth_rate=income_growth_pct / 100,
        income_growth_range=income_vol_pct / 100,
        current_savings=float(current_savings),
        savings_rate=savings_rate_pct / 100,
        other_savings=float(other_savings),
        ss_annual_benefit=ss_benefit,
        run=run,
    )


def _welcome_screen():
    """Landing screen shown before any simulation is run."""
    st.info(
        "Configure your financial profile in the **sidebar** and click "
        "**Run Analysis** to generate your personalized retirement outlook."
    )

    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("1. Input", "Enter age, income, savings, and goals in the sidebar."),
        ("2. Optimize", "An optimization algorithm tests thousands of possible investment mixes "
                        "to find the allocation most likely to carry you through retirement."),
        ("3. Simulate", f"{SIMULATION_COUNT:,} Monte Carlo paths model both your "
                        "accumulation and retirement drawdown phases."),
        ("4. Visualise", "Interactive charts reveal projected outcomes, "
                         "scenario trade-offs, and downside risks."),
    ]
    for col, (title, desc) in zip([col1, col2, col3, col4], steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.divider()

    st.subheader("Asset Class Universe")
    st.caption(
        "The model draws from six asset classes calibrated on data from 1926 to 2023. "
        "Bubble size encodes the Sharpe ratio, a measure of risk-adjusted return showing "
        "how much return an asset historically delivered per unit of risk taken. "
        "Hover over each asset for full statistics."
    )
    st.plotly_chart(fig_risk_return(), use_container_width=True)

    st.subheader("Asset-Return Correlation Matrix")
    st.caption(
        "The model uses correlated multi-asset returns (not independent draws), "
        "which correctly captures diversification benefits and their limits during crises. "
        "Red = positive correlation, Blue = negative (diversifying) correlation."
    )
    st.plotly_chart(fig_correlation_heatmap(), use_container_width=True)


def _run_analysis(inputs: dict):
    """Run all computations and store results in st.session_state."""
    cfg = build_config(**{k: v for k, v in inputs.items() if k != "run"})

    progress = st.progress(0, text="Optimizing portfolio allocation…")
    optimal_weights = optimize_portfolio(cfg)
    progress.progress(33, text=f"Running {SIMULATION_COUNT:,}-path accumulation simulation…")

    savings_results, income_results, wd_results, goal_proxy, paths = run_simulations(
        optimal_weights, cfg, SIMULATION_COUNT, return_paths=True
    )
    proxy_rate = float(goal_proxy.mean() * 100)

    progress.progress(66, text=f"Simulating {cfg.retirement_years}-year withdrawal phase…")
    survived, survival_curve = run_withdrawal_simulation(
        savings_results, wd_results,
        CONSERVATIVE_RETIREMENT_WEIGHTS, cfg.retirement_years,
        return_survival_curve=True,
    )
    goal_rate = float(survived.mean() * 100)

    progress.progress(80, text="Running scenario & sensitivity analysis…")
    scenarios = {}
    orig_sr = cfg.savings_rate
    for inc in (0.01, 0.05, 0.10):
        new_rate = min(1.0, orig_sr + inc)
        scenarios[f"+{int(inc * 100)} percentage point savings rate → {new_rate:.0%}"] = run_scenario(
            cfg, optimal_weights, sr_override=new_rate
        )
    for extra in (2, 5):
        scenarios[f"Retire {extra} yrs later (age {cfg.retirement_age + extra})"] = run_scenario(
            cfg, optimal_weights, extra_years=extra
        )

    sens_m1 = run_sensitivity(cfg, optimal_weights, -0.01)
    sens_m2 = run_sensitivity(cfg, optimal_weights, -0.02)

    progress.progress(100, text="Done.")
    progress.empty()

    st.session_state["results"] = dict(
        cfg=cfg,
        optimal_weights=optimal_weights,
        savings_results=savings_results,
        income_results=income_results,
        wd_results=wd_results,
        paths=paths,
        survived=survived,
        survival_curve=survival_curve,
        proxy_rate=proxy_rate,
        goal_rate=goal_rate,
        scenarios=scenarios,
        sens_m1=sens_m1,
        sens_m2=sens_m2,
    )


def _display_results(r: dict):
    """Render the full results dashboard from stored session state."""
    cfg:  SimConfig  = r["cfg"]
    wts:  np.ndarray = r["optimal_weights"]
    rate: float      = r["goal_rate"]

    # ── Banner ────────────────────────────────────────────────────────────────
    col_kpi, col_msg = st.columns([1, 3])

    with col_kpi:
        delta_text = f"Pre-retirement check (optimistic estimate): {r['proxy_rate']:.1f}%"
        st.metric(
            label="Withdrawal Survival Rate",
            value=f"{rate:.1f}%",
            delta=delta_text,
            help=(
                f"Fraction of {SIMULATION_COUNT:,} simulated retirement paths where "
                f"your portfolio lasts the full {cfg.retirement_years}-year retirement. "
                "The pre-retirement check is an optimistic upper bound; "
                "withdrawal survival is the more conservative and realistic figure."
            ),
        )

    with col_msg:
        label = success_label(rate)
        if rate >= HIGH_SUCCESS_RATE:
            st.success(f"**{label}**: Your plan has a strong probability of supporting you through retirement.")
        elif rate >= DEFAULT_TARGET_RATE:
            st.success(f"**{label}**: You are likely on track under these assumptions.")
        elif rate >= MIN_SUCCESS_RATE:
            st.warning(f"**{label}**: Meaningful chance of running short. Review the recommendations below.")
        else:
            st.error(f"**{label}**: Portfolio is likely to run out before the end of retirement. Significant changes are needed.")

    # ── Summary metrics ───────────────────────────────────────────────────────
    p10  = float(np.percentile(r["savings_results"], 10))
    p50  = float(np.median(r["savings_results"]))
    p90  = float(np.percentile(r["savings_results"], 90))
    wd50 = float(np.median(r["wd_results"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Median Nest Egg at Retirement",
        format_currency(p50),
        help=f"Median simulated portfolio value at age {cfg.retirement_age}.",
    )
    c2.metric(
        "Bad-Luck Scenario (10th Percentile)",
        format_currency(p10),
        delta=format_currency(p10 - p50),
        delta_color="inverse",
        help="Bottom 10th percentile of simulated outcomes: a poor-luck scenario.",
    )
    c3.metric(
        "Good-Luck Scenario (90th Percentile)",
        format_currency(p90),
        delta=format_currency(p90 - p50),
        help="Top 90th percentile of simulated outcomes: a favorable-luck scenario.",
    )
    c4.metric(
        "Median Annual Withdrawal",
        format_currency(wd50) + "/yr",
        help=(
            "Median first-year draw from your portfolio in retirement, after "
            "Social Security. Grows with inflation every year."
        ),
    )

    st.divider()

    # ── Savings fan chart ─────────────────────────────────────────────────────
    st.subheader("Projected Portfolio Trajectory")
    st.caption(
        "Simulated portfolio value from today through your target retirement age. "
        "Shaded bands show the 5th to 95th, 10th to 90th, and 25th to 75th percentile ranges across all paths. "
        "The corridor widens over time as market and income uncertainty compounds; "
        "a well-funded plan keeps even the P10 band (bad luck) above your required nest egg."
    )
    st.plotly_chart(fig_savings_fan(r["paths"], cfg), use_container_width=True)

    # ── Portfolio & glidepath ─────────────────────────────────────────────────
    st.subheader("Optimal Portfolio Allocation & Glidepath")
    st.caption(
        "**Left:** Optimized starting allocation, maximizing withdrawal survival within hard constraints: "
        "alternative investments are capped at 10% and bonds plus cash must meet a minimum safety floor. "
        "**Right:** How the portfolio automatically shifts toward safer investments over time, "
        "blending from the growth allocation today toward a conservative income profile at retirement — "
        "similar to a target-date fund, which automatically becomes more conservative as your retirement date approaches."
    )
    col_d, col_g = st.columns(2)
    with col_d:
        st.plotly_chart(fig_portfolio_donut(wts, cfg.current_savings), use_container_width=True)
    with col_g:
        st.plotly_chart(fig_glidepath(wts, cfg), use_container_width=True)

    # ── Optimal portfolio detail table ────────────────────────────────────────
    with st.expander("Optimal Portfolio Detail"):
        bond_names = ("Corporate Bonds", "Government Bonds", "Cash")
        alt_names  = ("Real Estate (REIT)", "Commodities")

        rows = [{
            "Asset Class":     name,
            "Weight":          f"{w * 100:.1f}%",
            "Amount (today)":  f"${cfg.current_savings * w:,.0f}",
            "Exp. Return":     f"{ASSET_CLASSES[name]['expected_return'] * 100:.1f}%",
            "Annual Volatility":  f"{ASSET_CLASSES[name]['stdev'] * 100:.1f}%",
        } for name, w in zip(ASSET_NAMES, wts)]

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        bond_total = sum(wts[ASSET_NAMES.index(a)] for a in bond_names if a in ASSET_NAMES)
        alt_total  = sum(wts[ASSET_NAMES.index(a)] for a in alt_names  if a in ASSET_NAMES)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "Bond + Cash Total", f"{bond_total:.1%}",
            help=f"Minimum required: {cfg.min_bond:.0%}, based on the glidepath floor for a {cfg.years_to_simulate}-year horizon.",
        )
        mc2.metric(
            "Alternatives (Real Estate Investment Trusts + Commodities)", f"{alt_total:.1%}",
            help=f"Hard cap: {ALTERNATIVES_MAX_SHARE:.0%} of the portfolio.",
        )
        mc3.metric("Years to Retirement", cfg.years_to_simulate)

    # ── Withdrawal survival curve ─────────────────────────────────────────────
    st.subheader("Withdrawal Phase: Survival Probability")
    st.caption(
        "Each year of retirement some simulation paths exhaust their portfolio. "
        "This chart shows what fraction survive through each year. "
        "Withdrawals grow at 2% per year to keep pace with inflation. "
        "A steep early decline signals high **sequence-of-returns risk**: "
        "a severe market downturn in the first few retirement years can permanently impair "
        "a portfolio even if the nest egg appeared adequate at the retirement date."
    )
    st.plotly_chart(fig_withdrawal_survival(r["survival_curve"], cfg), use_container_width=True)

    # ── Annual withdrawal distribution ────────────────────────────────────────
    st.subheader("Distribution of Required Annual Withdrawals")
    st.caption(
        "Histogram of the first-year portfolio draw (post Social Security) across all simulated paths. "
        "The spread reflects uncertainty in your final income; a higher-income path "
        "requires more income replacement in retirement. "
        "Plan for at least the P75 withdrawal to build a margin of safety."
    )
    st.plotly_chart(fig_withdrawal_distribution(r["wd_results"]), use_container_width=True)

    # ── Scenario analysis ─────────────────────────────────────────────────────
    st.subheader("Scenario Analysis")
    st.caption(
        "Each bar estimates the withdrawal survival rate under a single modified assumption, "
        "with everything else held constant. "
        "Increasing your savings rate by even 1 to 5 percentage points "
        "or deferring retirement by a few years can meaningfully shift your odds."
    )
    st.plotly_chart(fig_scenarios(rate, r["scenarios"]), use_container_width=True)

    # ── Sensitivity to lower returns ──────────────────────────────────────────
    st.subheader("Sensitivity to Lower Future Returns")
    st.caption(
        "Historical returns (1926 to 2023) may not repeat. "
        "These bars show how success degrades if future asset returns are "
        "structurally lower across all asset classes: "
        "a scenario consistent with lower long-run productivity growth "
        "or persistently higher valuations."
    )
    st.plotly_chart(
        fig_sensitivity(rate, r["sens_m1"], r["sens_m2"]),
        use_container_width=True,
    )

    # ── Recommendations ───────────────────────────────────────────────────────
    st.subheader("Recommendations")
    if rate < DEFAULT_TARGET_RATE:
        st.warning(
            "Your current plan falls below the 75% success target. "
            "The scenarios above show quantified estimates for each action below:"
        )
        st.markdown("""
* Increase your savings rate. Even +1 percentage point has a measurable effect; +5 to 10 percentage points is transformative.
* Delay retirement. Each additional year adds contributions and compounds existing savings.
* Review your spending target. A lower income replacement rate reduces the required nest egg.
* Verify your Social Security entitlement. Log in at [ssa.gov/myaccount](https://www.ssa.gov/myaccount/) to see your personalized benefit estimate; it may be higher than assumed here.
* Consult a fee-only financial planner. A NAPFA planner can account for tax treatment, real estate equity, and other assets not modeled here. See napfa.org to find one.
        """)
    else:
        st.success("You are on track. To build a wider margin of safety:")
        st.markdown("""
* Re-run this analysis annually as your income, savings, and goals evolve.
* Consider modest savings-rate increases to widen your buffer against lower-return scenarios.
* Review your asset allocation annually. The glidepath is modeled as a smooth blend, but in practice periodic rebalancing is required.
        """)

    # ── Model caveats ─────────────────────────────────────────────────────────
    with st.expander("Model Assumptions & Caveats"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**Modeling choices**
* The glidepath is **linear and deterministic**, not dynamically optimized. A full multi-period optimization would better capture life-cycle shifts.
* Income autocorrelation is fixed at **0.30**, following the Guvenen 2009 literature average. Your career may be more or less persistent.
* The Social Security cost-of-living adjustment is assumed at 2% per year; actual adjustments vary.
* Tax treatment is not modeled. All figures are pre-tax.
* The withdrawal model assumes a fixed inflation-adjusted draw each year. It does not model flexible spending strategies such as dynamic withdrawals or guardrail methods.
            """)
        with col_b:
            st.markdown("""
**Data & assumptions**
* Return data covers **1926 to 2023** across six asset classes: S&P 500, Bloomberg US Corporate IG, NAREIT All REITs, S&P GSCI, 10-year Treasuries, and T-Bills. Past returns do not guarantee future results.
* Returns are modeled as correlated log-normal draws using a Cholesky decomposition of the asset covariance matrix. Correlations are long-run estimates; they fluctuate, especially during crises.
* Life expectancy follows simplified SSA actuarial tables; individual health circumstances vary widely.
* The optimizer uses Differential Evolution, a global search algorithm, running 500 simulations per candidate allocation. Final results use 5,000 simulation paths.
* Consult a **fee-only financial planner** at napfa.org for personalized advice.
            """)



# =============================================================================
# Entry Point
# =============================================================================

def main():
    st.title("Retirement Investment Simulator")
    st.markdown(
        "This is both a simulation and an optimization model. "
        "It optimizes your portfolio allocation across six asset classes "
        "and then projects, across thousands of simulated market futures, "
        "whether your savings are likely to last through retirement. "
        "All inputs are in the sidebar."
    )

    inputs = _sidebar_inputs()

    if inputs["run"]:
        _run_analysis(inputs)

    if "results" not in st.session_state:
        _welcome_screen()
    else:
        _display_results(st.session_state["results"])


main()
