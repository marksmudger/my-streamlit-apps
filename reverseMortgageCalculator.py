import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Home Affordability Calculator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 5px solid #2563eb;
        margin-bottom: 8px;
    }
    .metric-card h2 { margin: 0 0 4px 0; font-size: 2rem; color: #1e3a5f; }
    .metric-card p  { margin: 0; color: #6b7280; font-size: 0.9rem; }
    .tip-box {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 0.88rem;
        color: #78350f;
        line-height: 1.6;
    }
    .error-box {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 10px;
        padding: 16px 20px;
        color: #991b1b;
    }
    .dti-good { color: #166534; background: #dcfce7; border: 1px solid #86efac; border-radius: 8px; padding: 10px 16px; margin-top: 8px; }
    .dti-warn { color: #92400e; background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 10px 16px; margin-top: 8px; }
    .dti-high { color: #991b1b; background: #fee2e2; border: 1px solid #fca5a5; border-radius: 8px; padding: 10px 16px; margin-top: 8px; }
    .pmi-note { background: #fefce8; border: 1px solid #fcd34d; border-radius: 8px; padding: 10px 16px; font-size: 0.87rem; color: #78350f; margin-top: 6px; }
    .closing-note { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 16px; font-size: 0.87rem; color: #14532d; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Core calculation ──────────────────────────────────────────────────────────
@st.cache_data
def calculate_max_home_price(
    target_monthly_payment,
    down_payment,
    interest_rate=7.0,
    property_tax_rate=1.2,
    monthly_insurance=150,
    monthly_hoa=0,
    loan_term_years=30,
    pmi_annual_rate=0.80,  # % of loan amount per year; auto-applied when down < 20%
):
    monthly_rate = interest_rate / 100 / 12
    monthly_tax_rate = property_tax_rate / 100 / 12
    num_payments = loan_term_years * 12

    available = target_monthly_payment - monthly_insurance - monthly_hoa

    if available <= 0:
        return {
            "max_home_price": 0,
            "max_loan_amount": 0,
            "down_payment": down_payment,
            "down_payment_percent": 0,
            "pmi_monthly": 0,
            "total_interest": 0,
            "monthly_breakdown": {
                "principal_interest": 0,
                "property_tax": 0,
                "insurance": monthly_insurance,
                "hoa": monthly_hoa,
                "pmi": 0,
                "total": target_monthly_payment,
            },
            "loan_details": {
                "interest_rate": interest_rate,
                "term_years": loan_term_years,
                "property_tax_rate": property_tax_rate,
            },
            "error": "Your target payment is too low to cover fixed monthly costs (insurance and HOA).",
        }

    if monthly_rate > 0:
        factor = (1 + monthly_rate) ** num_payments
        mortgage_factor = (monthly_rate * factor) / (factor - 1)
    else:
        mortgage_factor = 1 / num_payments

    # First solve without PMI to determine whether it applies
    max_home_price = (available + down_payment * mortgage_factor) / (
        mortgage_factor + monthly_tax_rate
    )

    # Auto-apply PMI when down payment < 20% of the solved home price
    pmi_monthly_rate = 0.0
    if pmi_annual_rate > 0 and down_payment < 0.20 * max_home_price:
        pmi_monthly_rate = pmi_annual_rate / 100 / 12
        # Re-solve: PMI is proportional to the loan amount, so it adds a term on the loan portion
        max_home_price = (available + down_payment * (mortgage_factor + pmi_monthly_rate)) / (
            mortgage_factor + pmi_monthly_rate + monthly_tax_rate
        )

    max_loan_amount = max_home_price - down_payment

    # Edge case: down payment alone exceeds the calculated home price
    if max_loan_amount <= 0:
        property_tax = max_home_price * monthly_tax_rate
        return {
            "max_home_price": max_home_price,
            "max_loan_amount": 0,
            "down_payment": down_payment,
            "down_payment_percent": 100.0,
            "pmi_monthly": 0,
            "total_interest": 0,
            "monthly_breakdown": {
                "principal_interest": 0,
                "property_tax": property_tax,
                "insurance": monthly_insurance,
                "hoa": monthly_hoa,
                "pmi": 0,
                "total": property_tax + monthly_insurance + monthly_hoa,
            },
            "loan_details": {
                "interest_rate": interest_rate,
                "term_years": loan_term_years,
                "property_tax_rate": property_tax_rate,
            },
            "info": "Your down payment covers the full home price — no mortgage needed.",
        }

    down_payment_percent = (down_payment / max_home_price) * 100

    principal_interest = max_loan_amount * mortgage_factor
    property_tax = max_home_price * monthly_tax_rate
    pmi_monthly = max_loan_amount * pmi_monthly_rate
    total_payment = principal_interest + property_tax + monthly_insurance + monthly_hoa + pmi_monthly
    total_interest = (principal_interest * num_payments) - max_loan_amount

    return {
        "max_home_price": max_home_price,
        "max_loan_amount": max_loan_amount,
        "down_payment": down_payment,
        "down_payment_percent": down_payment_percent,
        "pmi_monthly": pmi_monthly,
        "total_interest": total_interest,
        "monthly_breakdown": {
            "principal_interest": principal_interest,
            "property_tax": property_tax,
            "insurance": monthly_insurance,
            "hoa": monthly_hoa,
            "pmi": pmi_monthly,
            "total": total_payment,
        },
        "loan_details": {
            "interest_rate": interest_rate,
            "term_years": loan_term_years,
            "property_tax_rate": property_tax_rate,
        },
    }


@st.cache_data
def build_amortization(loan_amount, annual_rate, term_years):
    """Return a DataFrame with yearly loan balance, cumulative principal, and interest paid."""
    monthly_rate = annual_rate / 100 / 12
    num_payments = term_years * 12
    if monthly_rate > 0:
        factor = (1 + monthly_rate) ** num_payments
        monthly_payment = loan_amount * (monthly_rate * factor) / (factor - 1)
    else:
        monthly_payment = loan_amount / num_payments

    balance = loan_amount
    cum_interest = 0.0
    cum_principal = 0.0
    rows = [{"Year": 0, "Remaining Balance": balance,
             "Cumulative Principal Paid": 0.0, "Cumulative Interest Paid": 0.0}]

    for month in range(1, num_payments + 1):
        interest = balance * monthly_rate
        principal = monthly_payment - interest
        balance -= principal
        cum_interest += interest
        cum_principal += principal
        if month % 12 == 0 or month == num_payments:
            rows.append(
                {
                    "Year": month // 12,
                    "Remaining Balance": max(balance, 0),
                    "Cumulative Principal Paid": cum_principal,
                    "Cumulative Interest Paid": cum_interest,
                }
            )

    return pd.DataFrame(rows)


# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏠 Your Numbers")
    st.caption("Adjust the sliders to see results update instantly.")

    st.subheader("Budget")
    target_payment = st.slider(
        "Monthly housing budget",
        min_value=500, max_value=15_000, value=3_500, step=50,
        format="$%d",
        help="The total amount you want to spend on housing each month, including all costs.",
    )
    down_payment = st.slider(
        "Down payment saved",
        min_value=0, max_value=500_000, value=60_000, step=1_000,
        format="$%d",
    )
    gross_monthly_income = st.slider(
        "Gross monthly income",
        min_value=1_000, max_value=50_000, value=10_000, step=500,
        format="$%d",
        help="Pre-tax household income per month. Used to show your housing debt-to-income (DTI) ratio.",
    )

    st.subheader("Loan")
    interest_rate = st.slider(
        "Interest rate (%)", min_value=2.0, max_value=12.0, value=7.0, step=0.05,
        format="%.2f%%",
    )
    loan_term = st.selectbox("Loan term", [30, 20, 15, 10], index=0, format_func=lambda x: f"{x} years")

    st.subheader("Other Monthly Costs")
    monthly_insurance = st.slider(
        "Home insurance", min_value=0, max_value=1_000, value=150, step=10,
        format="$%d",
    )
    property_tax_rate = st.slider(
        "Property tax rate (%)", min_value=0.0, max_value=4.0, value=1.2, step=0.05,
        format="%.2f%%",
    )
    monthly_hoa = st.slider(
        "HOA fees", min_value=0, max_value=2_000, value=0, step=25,
        format="$%d",
    )

    st.divider()
    st.subheader("Budget Comparison")
    scenario_increment = st.slider(
        "Step size between scenarios", min_value=100, max_value=1_000, value=250, step=50,
        format="$%d",
        help="The chart below will show this many dollars added per scenario.",
    )
    num_scenarios = st.slider("Number of scenarios", min_value=3, max_value=10, value=6)


# ── Run calculation ───────────────────────────────────────────────────────────
results = calculate_max_home_price(
    target_monthly_payment=target_payment,
    down_payment=down_payment,
    interest_rate=interest_rate,
    property_tax_rate=property_tax_rate,
    monthly_insurance=monthly_insurance,
    monthly_hoa=monthly_hoa,
    loan_term_years=loan_term,
)

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🏠 Home Affordability Calculator")
st.markdown(
    """<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;
        padding:18px 22px;margin-bottom:4px;color:#0c4a6e;font-size:0.97rem;line-height:1.8;">
    <b>What does this tool do?</b><br>
    Most mortgage calculators ask you to enter a home price and then tell you the monthly
    payment. This tool works the other way around: you enter the
    maximum monthly amount you are willing to spend on housing, and it calculates the
    highest home price you can realistically afford, accounting for your mortgage
    payment, property taxes, home insurance, and HOA fees all at once.<br><br>
    👈 Set your numbers in the left panel. The key results appear at the top of this
    page, followed by charts that break down your monthly payment, compare different budget
    levels, show how your loan balance shrinks over time, and reveal how sensitive your buying
    power is to changes in interest rates and down payment size.
    </div>""",
    unsafe_allow_html=True,
)

# ── Error state ───────────────────────────────────────────────────────────────
if "error" in results:
    st.markdown(
        f'<div class="error-box">⚠️ <b>Payment too low.</b> {results["error"]}<br>'
        f"Your budget of <b>${target_payment:,}/month</b> doesn't cover the fixed costs of "
        f"<b>${monthly_insurance + monthly_hoa:,}/month</b> (insurance + HOA). "
        f"Increase your budget or lower these fixed costs.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

breakdown = results["monthly_breakdown"]
loan_details = results["loan_details"]

# ── Hero metrics ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="metric-card">'
        f"<p>Maximum Home Price</p>"
        f"<h2>${results['max_home_price']:,.0f}</h2>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card">'
        f"<p>Loan Amount</p>"
        f"<h2>${results['max_loan_amount']:,.0f}</h2>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card">'
        f"<p>Down Payment</p>"
        f"<h2>${results['down_payment']:,.0f} <span style='font-size:1rem;color:#6b7280'>({results['down_payment_percent']:.1f}%)</span></h2>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col4:
    total_loan_cost = results["max_loan_amount"] + results.get("total_interest", 0)
    st.markdown(
        f'<div class="metric-card">'
        f"<p>Total Loan Cost</p>"
        f"<h2>${total_loan_cost:,.0f}</h2>"
        f"<p style='font-size:0.78rem;'>incl. ${results.get('total_interest', 0):,.0f} interest</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── DTI indicator ─────────────────────────────────────────────────────────────
front_end_dti = (breakdown["total"] / gross_monthly_income) * 100
if front_end_dti <= 28:
    dti_class = "dti-good"
    dti_icon = "✅"
    dti_verdict = "within the recommended 28% front-end DTI guideline"
elif front_end_dti <= 36:
    dti_class = "dti-warn"
    dti_icon = "⚠️"
    dti_verdict = "slightly above the 28% guideline — lenders may still approve, but consider reducing your budget"
else:
    dti_class = "dti-high"
    dti_icon = "🚨"
    dti_verdict = "above the 36% threshold — lenders may require compensating factors or deny the loan"

st.markdown(
    f'<div class="{dti_class}">'
    f"{dti_icon} <b>Housing DTI: {front_end_dti:.1f}%</b> — Your ${breakdown['total']:,.0f}/month payment is {dti_verdict}. "
    f"(28% of your ${gross_monthly_income:,}/month income = ${gross_monthly_income * 0.28:,.0f}/month.)"
    f"</div>",
    unsafe_allow_html=True,
)

# ── PMI notice ────────────────────────────────────────────────────────────────
if results["pmi_monthly"] > 0:
    st.markdown(
        f'<div class="pmi-note">'
        f"🔒 <b>PMI included:</b> Your down payment ({results['down_payment_percent']:.1f}%) is below 20%, so "
        f"Private Mortgage Insurance of <b>${results['pmi_monthly']:,.0f}/month</b> has been added to your payment. "
        f"PMI typically cancels once you reach 20% equity in the home."
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Closing cost estimate ─────────────────────────────────────────────────────
closing_low = results["max_home_price"] * 0.02
closing_high = results["max_home_price"] * 0.05
st.markdown(
    f'<div class="closing-note">'
    f"🏷️ <b>Estimated closing costs:</b> ${closing_low:,.0f}–${closing_high:,.0f} "
    f"(2–5% of home price). Budget for this separately — it's due at signing, on top of your down payment."
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()

# ── Tips ─────────────────────────────────────────────────────────────────────
st.subheader("📋 Things to Keep in Mind")
st.markdown(
    """
<div class="tip-box">
<b>Before you make an offer:</b>
<ul style="margin-top:8px;">
  <li>🏦 <b>Debt-to-Income (DTI):</b> Lenders typically want your total monthly debt payments (including this mortgage) to be under 43% of your gross monthly income.</li>
  <li>💸 <b>Closing costs:</b> Budget an extra <b>2–5%</b> of the home price for closing costs (e.g. $8,000–$20,000 on a $400,000 home).</li>
  <li>🔧 <b>Maintenance:</b> Plan for roughly <b>1–2%</b> of the home's value per year in upkeep and repairs.</li>
  <li>🎯 <b>Emergency fund:</b> Keep 3–6 months of expenses in savings even after closing.</li>
  <li>📈 <b>Rate changes:</b> This calculator uses a fixed rate. ARM loans can start lower but may rise significantly over time.</li>
</ul>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# ── Row 1: Payment donut + scenario bar ──────────────────────────────────────
chart_col1, chart_col2 = st.columns([1, 1.4])

# Donut: monthly payment breakdown
with chart_col1:
    st.subheader("Where does your monthly payment go?")

    labels = ["Principal & Interest", "Property Tax", "Home Insurance"]
    values = [
        breakdown["principal_interest"],
        breakdown["property_tax"],
        breakdown["insurance"],
    ]
    colors = ["#2563eb", "#10b981", "#f59e0b"]

    if breakdown["hoa"] > 0:
        labels.append("HOA Fees")
        values.append(breakdown["hoa"])
        colors.append("#8b5cf6")

    if breakdown.get("pmi", 0) > 0:
        labels.append("PMI")
        values.append(breakdown["pmi"])
        colors.append("#ec4899")

    donut = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker_colors=colors,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}/month<extra></extra>",
            textfont_size=13,
        )
    )
    donut.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>${breakdown['total']:,.0f}</b><br>/month",
                x=0.5, y=0.5,
                font_size=18,
                showarrow=False,
            )
        ],
        height=340,
    )
    st.plotly_chart(donut, use_container_width=True)

    # Tabular breakdown below chart
    breakdown_df = pd.DataFrame(
        {
            "Cost": labels,
            "Monthly Amount": [f"${v:,.0f}" for v in values],
        }
    )
    total_row = pd.DataFrame({"Cost": ["**Total**"], "Monthly Amount": [f"**${breakdown['total']:,.0f}**"]})
    breakdown_df = pd.concat([breakdown_df, total_row], ignore_index=True)
    st.dataframe(breakdown_df, hide_index=True, use_container_width=True)


# Bar: scenario comparison
with chart_col2:
    st.subheader("What if you adjusted your monthly budget?")
    st.caption(
        f"Each bar shows the max home price for a different monthly budget, "
        f"going up by ${scenario_increment:,} at a time."
    )

    half = num_scenarios // 2
    base = max(500, target_payment - half * scenario_increment)
    scenario_payments = [base + i * scenario_increment for i in range(num_scenarios)]

    scenario_rows = []
    for pmt in scenario_payments:
        r = calculate_max_home_price(
            target_monthly_payment=pmt,
            down_payment=down_payment,
            interest_rate=interest_rate,
            property_tax_rate=property_tax_rate,
            monthly_insurance=monthly_insurance,
            monthly_hoa=monthly_hoa,
            loan_term_years=loan_term,
        )
        if "error" not in r:
            scenario_rows.append(
                {"Monthly Budget": f"${pmt:,}", "Max Home Price": r["max_home_price"], "_payment": pmt}
            )

    scenario_df = pd.DataFrame(scenario_rows)

    bar_colors = [
        "#2563eb" if row["_payment"] == target_payment else "#93c5fd"
        for _, row in scenario_df.iterrows()
    ]

    max_home_price_vals = scenario_df["Max Home Price"].tolist()
    y_max = max(max_home_price_vals) * 1.18  # 18% headroom so outside labels never clip

    bar_fig = go.Figure(
        go.Bar(
            x=scenario_df["Monthly Budget"],
            y=max_home_price_vals,
            marker_color=bar_colors,
            text=[f"${v:,.0f}" for v in max_home_price_vals],
            textposition="outside",
            textfont=dict(size=12, color="#1e3a5f"),
            hovertemplate="Budget: <b>%{x}</b><br>Max Home Price: <b>$%{y:,.0f}</b><extra></extra>",
        )
    )
    bar_fig.update_layout(
        yaxis=dict(
            tickformat="$,.0f",
            title="Maximum Home Price",
            gridcolor="#e5e7eb",
            range=[0, y_max],
        ),
        xaxis_title="Monthly Housing Budget",
        plot_bgcolor="white",
        margin=dict(t=60, b=10, l=10, r=10),
        height=420,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

st.divider()

# ── Row 2: Amortization chart ────────────────────────────────────────────────
st.subheader("Where do your payments go over the life of the loan?")
st.caption(
    "The stacked areas show how your monthly payments are split: "
    "blue goes toward paying off the loan (principal), amber goes to the bank as interest. "
    "The red dashed line is how much you still owe at any point in time."
)

if results["max_loan_amount"] <= 0:
    st.info("No loan needed — your down payment covers the full home price.")
    st.stop()

amort_df = build_amortization(results["max_loan_amount"], interest_rate, loan_term)

amort_fig = go.Figure()

# Stacked area: cumulative principal paid (bottom)
amort_fig.add_trace(
    go.Scatter(
        x=amort_df["Year"],
        y=amort_df["Cumulative Principal Paid"],
        name="Principal Paid",
        stackgroup="paid",
        fillcolor="rgba(37, 99, 235, 0.55)",
        line=dict(color="#2563eb", width=1),
        hovertemplate="Year %{x}<br>Principal paid so far: <b>$%{y:,.0f}</b><extra></extra>",
    )
)

# Stacked area: cumulative interest paid (top)
amort_fig.add_trace(
    go.Scatter(
        x=amort_df["Year"],
        y=amort_df["Cumulative Interest Paid"],
        name="Interest Paid",
        stackgroup="paid",
        fillcolor="rgba(245, 158, 11, 0.55)",
        line=dict(color="#f59e0b", width=1),
        hovertemplate="Year %{x}<br>Interest paid so far: <b>$%{y:,.0f}</b><extra></extra>",
    )
)

# Remaining balance as a separate reference line
amort_fig.add_trace(
    go.Scatter(
        x=amort_df["Year"],
        y=amort_df["Remaining Balance"],
        name="Remaining Balance",
        line=dict(color="#ef4444", width=2.5, dash="dot"),
        hovertemplate="Year %{x}<br>Still owe: <b>$%{y:,.0f}</b><extra></extra>",
    )
)

# Annotate the halfway point
midpoint_year = loan_term // 2
mid_rows = amort_df[amort_df["Year"] == midpoint_year]
if not mid_rows.empty:
    mid_row = mid_rows.iloc[0]
    amort_fig.add_annotation(
        x=mid_row["Year"],
        y=mid_row["Remaining Balance"],
        text=f"Year {midpoint_year}: ${mid_row['Remaining Balance']:,.0f} still owed",
        showarrow=True,
        arrowhead=2,
        ax=70, ay=-40,
        font=dict(size=11, color="#991b1b"),
        bgcolor="white",
        bordercolor="#ef4444",
        borderwidth=1,
    )

total_interest = amort_df["Cumulative Interest Paid"].iloc[-1]
amort_fig.update_layout(
    xaxis=dict(title="Year into Loan", dtick=5, gridcolor="#e5e7eb"),
    yaxis=dict(tickformat="$,.0f", gridcolor="#e5e7eb"),
    plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="left", x=0),
    margin=dict(t=50, b=20),
    height=380,
    hovermode="x unified",
)
st.plotly_chart(amort_fig, use_container_width=True)

# Interest cost callout
# Note: st.info() can misparse "$" as LaTeX delimiters, so we use HTML instead.
principal = results["max_loan_amount"]
total_paid = principal + total_interest
interest_pct = (total_interest / principal * 100) if principal > 0 else 0

# Yes, ~140% interest over 30 years at 7% is mathematically correct.
# Monthly mortgage payments are front-loaded with interest (amortisation), so
# in the early years almost nothing reduces the principal balance.
st.markdown(
    f"""<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
        padding:16px 20px;color:#1e40af;font-size:0.95rem;line-height:1.7;">
    💡 Over {loan_term} years at {interest_rate:.2f}% you'll pay
    ${total_interest:,.0f} in interest on top of your ${principal:,.0f} loan,
    bringing the total cost of borrowing to ${total_paid:,.0f}
    ({interest_pct:.0f}% of what you originally borrowed).<br>
    <span style="font-size:0.85rem;color:#3b82f6;">
    ℹ️ This is expected. At {interest_rate:.2f}%, early payments are mostly interest with very
    little going toward principal. Making even one extra payment per year can shave years off
    your loan and save tens of thousands in interest.
    </span>
    </div>""",
    unsafe_allow_html=True,
)

st.divider()

# ── Row 3: Sensitivity analyses ──────────────────────────────────────────────
st.subheader("How do rate changes and your down payment affect affordability?")
st.caption(
    "These charts vary one factor at a time while keeping everything else the same, "
    "so you can see exactly what moves the needle on your maximum home price."
)

sens_col1, sens_col2 = st.columns(2)

# Chart: interest rate sensitivity
with sens_col1:
    st.markdown("**Interest Rate vs. Max Home Price**")
    st.caption(
        f"Your current rate of {interest_rate:.2f}% is marked in red. "
        "Notice how a 1-point change in rate can shift your price ceiling by tens of thousands."
    )

    rate_steps = [round(2.0 + i * 0.25, 2) for i in range(41)]  # 2.00 % → 12.00 %
    rate_rows = []
    for r in rate_steps:
        res = calculate_max_home_price(
            target_monthly_payment=target_payment,
            down_payment=down_payment,
            interest_rate=r,
            property_tax_rate=property_tax_rate,
            monthly_insurance=monthly_insurance,
            monthly_hoa=monthly_hoa,
            loan_term_years=loan_term,
        )
        if "error" not in res:
            rate_rows.append({"Rate": r, "Max Home Price": res["max_home_price"]})

    rate_df = pd.DataFrame(rate_rows)

    rate_fig = go.Figure()
    # Shade the region to the left of the current rate (lower = better)
    better_rates = rate_df[rate_df["Rate"] <= interest_rate]
    rate_fig.add_trace(
        go.Scatter(
            x=better_rates["Rate"],
            y=better_rates["Max Home Price"],
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.10)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    rate_fig.add_trace(
        go.Scatter(
            x=rate_df["Rate"],
            y=rate_df["Max Home Price"],
            line=dict(color="#2563eb", width=2.5),
            showlegend=False,
            hovertemplate="Rate: <b>%{x:.2f}%</b><br>Max home price: <b>$%{y:,.0f}</b><extra></extra>",
        )
    )
    rate_fig.add_vline(
        x=interest_rate,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"Your rate ({interest_rate:.2f}%)",
        annotation_position="top right",
        annotation_font=dict(color="#ef4444", size=11),
    )
    rate_fig.update_layout(
        xaxis=dict(title="Interest Rate", ticksuffix="%", gridcolor="#e5e7eb", dtick=1),
        yaxis=dict(title="Max Home Price", tickformat="$,.0f", gridcolor="#e5e7eb"),
        plot_bgcolor="white",
        margin=dict(t=20, b=20, l=10, r=10),
        height=350,
        hovermode="x",
    )
    st.plotly_chart(rate_fig, use_container_width=True)

# Chart: down payment sensitivity
with sens_col2:
    st.markdown("**Down Payment vs. Max Home Price**")
    st.caption(
        f"Your current down payment of ${down_payment:,} is marked in red. "
        "Saving more up front directly raises the home price you can afford."
    )

    dp_max = min(500_000, max(down_payment * 4, 200_000))
    dp_steps = [round(dp_max * i / 40) for i in range(41)]
    dp_rows = []
    for dp in dp_steps:
        res = calculate_max_home_price(
            target_monthly_payment=target_payment,
            down_payment=dp,
            interest_rate=interest_rate,
            property_tax_rate=property_tax_rate,
            monthly_insurance=monthly_insurance,
            monthly_hoa=monthly_hoa,
            loan_term_years=loan_term,
        )
        if "error" not in res:
            dp_rows.append({"Down Payment": dp, "Max Home Price": res["max_home_price"]})

    dp_df = pd.DataFrame(dp_rows)

    dp_fig = go.Figure()
    dp_fig.add_trace(
        go.Scatter(
            x=dp_df["Down Payment"],
            y=dp_df["Max Home Price"],
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
            line=dict(color="#2563eb", width=2.5),
            showlegend=False,
            hovertemplate="Down payment: <b>$%{x:,.0f}</b><br>Max home price: <b>$%{y:,.0f}</b><extra></extra>",
        )
    )
    ann_pos = "top right" if down_payment < dp_max * 0.65 else "top left"
    dp_fig.add_vline(
        x=down_payment,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"Your down payment (${down_payment:,})",
        annotation_position=ann_pos,
        annotation_font=dict(color="#ef4444", size=11),
    )
    dp_fig.update_layout(
        xaxis=dict(title="Down Payment", tickformat="$,.0f", gridcolor="#e5e7eb"),
        yaxis=dict(title="Max Home Price", tickformat="$,.0f", gridcolor="#e5e7eb"),
        plot_bgcolor="white",
        margin=dict(t=20, b=20, l=10, r=10),
        height=350,
        hovermode="x",
    )
    st.plotly_chart(dp_fig, use_container_width=True)

st.caption(
    "This calculator is for educational purposes only and does not constitute financial advice. "
    "Consult a licensed mortgage professional before making any home purchase decision."
)
