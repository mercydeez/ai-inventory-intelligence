"""Streamlit dashboard | Atharva Soundankar | as25dxb020 | MAIB | AI in Operations, Session II
Run: streamlit run app.py
"""
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import inventory as inv

NAVY, TEAL, BLUE, AMBER, RED, MUTED, GRID = "#0f1f3d", "#0e8f8a", "#3b6fd4", "#d98e04", "#c8402f", "#6b7a90", "#e6e9ef"
SAMPLE = Path(__file__).parent / "data" / "weekly_demand.csv"
SAMPLE_NAME = "Synthetic sample: SKU-A101, 104 weeks, seed 42"

st.set_page_config(page_title="AI-Powered Inventory Intelligence | Atharva Soundankar", page_icon="📦", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; color: {NAVY}; }}
.stApp {{ background: #faf8f4; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
section[data-testid="stSidebar"] {{ background: #f3f0e9; border-right: 1px solid #e4dfd3; }}
.eyebrow {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: {TEAL}; font-weight: 600; }}
h1.hero {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.5rem; margin: .1rem 0 .2rem; color: {NAVY}; letter-spacing: -.01em; }}
.sub {{ color: {MUTED}; font-size: .92rem; margin-bottom: 1.3rem; }}
.kpi {{ background: #fff; border: 1px solid {GRID}; border-radius: 14px; padding: 1.05rem 1.2rem; height: 100%; }}
.kpi .l {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; color: {MUTED}; font-weight: 600; }}
.kpi .v {{ font-family: 'Fraunces', serif; font-size: 2.1rem; font-weight: 600; line-height: 1.15; margin: .25rem 0 .1rem; }}
.kpi .n {{ font-size: .78rem; color: {MUTED}; }}
.note {{ background: #fff; border: 1px solid {GRID}; border-left: 3px solid {TEAL}; border-radius: 8px; padding: .75rem 1rem; font-size: .86rem; color: #34425a; margin: .4rem 0 1rem; }}
.note.warn {{ border-left-color: {AMBER}; }}
h3 {{ font-weight: 600 !important; letter-spacing: -.01em; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {GRID}; }}
.stTabs [data-baseweb="tab"] {{ padding: .6rem 0; font-weight: 500; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {NAVY}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {TEAL}; }}
.stButton > button {{ background: {NAVY}; color: #fff; border: 0; border-radius: 10px; padding: .55rem 1.3rem; font-weight: 600; }}
.stButton > button:hover {{ background: {TEAL}; color: #fff; }}
.stButton > button p, .stButton > button:hover p {{ color: #fff !important; }}
.foot {{ color: {MUTED}; font-size: .76rem; margin-top: 2.2rem; border-top: 1px solid {GRID}; padding-top: .8rem; }}
</style>""", unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=30, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
                      font=dict(family="Inter", color=NAVY, size=12), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0), colorway=[NAVY, TEAL, BLUE, AMBER, RED])
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    return fig


def kpi(col, label, value, note, color=NAVY):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v" style="color:{color}">{value}</div>'
                 f'<div class="n">{note}</div></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def run_forecast(csv, horizon):
    return inv.forecast(inv.validate(pd.read_csv(io.StringIO(csv))), horizon)


@st.cache_data(show_spinner=False)
def run_risk(d, sd, L, sL, service, n):
    return inv.risk_scenarios(d, sd, L, sL, service, n)


@st.cache_data(show_spinner=False)
def run_replay(csv, L, sL, service, shock, weeks):
    return inv.adaptive_replay(inv.validate(pd.read_csv(io.StringIO(csv))), L, sL, service, shock, weeks)


@st.cache_data(show_spinner=False)
def run_sens(d, sd, L, sL, service, rop, trials):
    return inv.ai_sensitivity(d, sd, L, sL, service, rop, trials)


@st.cache_data(show_spinner=False)
def run_opt(**kw):
    return inv.optimize_rop(**kw)


# ---------- Sidebar: data and assumptions ----------
if "csv" not in st.session_state:
    st.session_state.csv, st.session_state.source = SAMPLE.read_text(), SAMPLE_NAME
with st.sidebar:
    st.markdown("### Data")
    up = st.file_uploader("Replace with your CSV (week, demand, optional promo, sku)", type="csv")
    if up is not None and st.session_state.get("up_name") != up.name:
        try:
            text = up.getvalue().decode("utf-8")
            inv.validate(pd.read_csv(io.StringIO(text)))
            st.session_state.csv, st.session_state.source, st.session_state.up_name = text, f"Uploaded: {up.name}", up.name
        except Exception as e:
            st.error(f"Upload rejected: {e}")
    if st.button("Reset to sample data"):
        st.session_state.csv, st.session_state.source = SAMPLE.read_text(), SAMPLE_NAME
        st.session_state.pop("up_name", None)
    st.caption(f"Source: {st.session_state.source}")
    horizon = st.slider("Forecast horizon (weeks)", 4, 26, 8)

F = run_forecast(st.session_state.csv, horizon)
sig = hash((st.session_state.source, horizon, len(st.session_state.csv)))  # reset defaults when data changes

with st.sidebar:
    st.markdown("### Inventory assumptions")
    st.caption("Demand defaults come from the forecast. Lead time and costs are assumptions, not measured records.")
    d = st.number_input("Mean weekly demand D̄ (units)", 1.0, 1e6, round(F["mean_forecast"], 1), 1.0, key=f"d{sig}")
    sd = st.number_input("Demand std dev σD (units/week)", 0.0, 1e5, round(F["sigma"], 1), 1.0, key=f"sd{sig}")
    L = st.number_input("Mean lead time L (weeks)", 0.5, 20.0, 2.0, 0.5)
    sL = st.number_input("Lead time std dev σL (weeks)", 0.0, 10.0, 0.5, 0.1)
    service = st.slider("Target service level", 0.80, 0.995, 0.95, 0.005)
    n_sim = st.select_slider("Monte Carlo runs", [500, 1000, 2000, 5000, 10000], 2000)
    st.markdown("### What-if: demand change")
    st.caption("Injected into the most recent weeks. Every AI tab reacts. Set to 0 for stable demand.")
    shock = st.slider("Demand change (%)", -50, 100, 40, 5) / 100
    shock_w = st.slider("Applied to the last N weeks", 4, 20, 10)

SS = inv.safety_stock(d, sd, L, sL, service)
R = run_risk(d, sd, L, sL, service, n_sim)
base = R["scenarios"][0]

# ---------- Header and KPIs ----------
st.markdown('<div class="eyebrow">AI-Powered Inventory Intelligence &amp; Reorder Point Optimization</div>'
            '<h1 class="hero">Welcome back, Atharva</h1>'
            '<div class="sub">Atharva Soundankar | as25dxb020 | MAIB | September 2025 &nbsp;·&nbsp; '
            'AI in Operations, Session II &nbsp;·&nbsp; SP Jain School of Global Management</div>', unsafe_allow_html=True)
risk_col = RED if base["stockout"] > 1 - service + 0.02 else (AMBER if base["stockout"] > 1 - service else TEAL)
k = st.columns(4)
kpi(k[0], "Forecast demand", f"{F['mean_forecast']:,.0f}", f"units per week, next {horizon} weeks")
kpi(k[1], "Safety stock", f"{SS['ss']:,.0f}", f"Z = {SS['z']:.3f} at {service:.1%} service")
kpi(k[2], "Reorder point", f"{SS['rop']:,.0f}", f"{d:,.0f} × {L:g} + safety stock")
kpi(k[3], "Stockout risk", f"{base['stockout']:.1%}", f"simulated, {n_sim:,} runs (target ≤ {1 - service:.1%})", risk_col)
st.write("")

tabB, tab1, tab2, tab3, tab4, tabA, tab5 = st.tabs(["Before vs With AI", "Demand forecast", "Safety stock and risk",
                                                    "Reorder optimization", "Adaptive intelligence", "Autopilot", "AI sensitivity"])

# ---------- A. Demand forecast ----------
with tab1:
    s = pd.DataFrame(F["series"])
    fig = go.Figure()
    band = s.dropna(subset=["band"])
    fig.add_trace(go.Scatter(x=list(band.week) + list(band.week[::-1]),
                             y=[b[1] for b in band.band] + [b[0] for b in band.band][::-1],
                             mode="lines", fill="toself", fillcolor="rgba(14,143,138,.13)", line=dict(width=0),
                             name="95% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(mode="lines", x=s.week, y=s.actual, name="Actual demand", line=dict(color=NAVY, width=2)))
    fig.add_trace(go.Scatter(mode="lines", x=s.week, y=s.baseline, name="Baseline (holdout)", line=dict(color=AMBER, width=2, dash="dot")))
    fig.add_trace(go.Scatter(mode="lines", x=s.week, y=s.model, name="Best model (holdout)", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(mode="lines", x=s.week, y=s.forecast, name="Forecast", line=dict(color=TEAL, width=3)))
    fig.add_shape(type="line", x0=F["split_week"], x1=F["split_week"], y0=0, y1=1, yref="paper",
                  line=dict(color=MUTED, dash="dash", width=1))
    fig.add_annotation(x=F["split_week"], y=1, yref="paper", text="train | test", showarrow=False,
                       xanchor="left", font=dict(color=MUTED, size=11))
    c1, c2 = st.columns([2.4, 1])
    with c1:
        st.markdown("### Historical demand vs forecast")
        st.plotly_chart(style(fig, 400), use_container_width=True)
    with c2:
        st.markdown("### Model comparison")
        m = pd.DataFrame(F["models"]).rename(columns={"name": "Model", "mae": "MAE", "wape": "WAPE %"})
        m["Model"] = m["Model"].str.replace("Linear regression: ", "LR: ")
        short = {"4-week moving average (baseline)": "Moving avg", "Linear regression: trend + promo": "Trend + promo",
                 "Linear regression: trend + promo + seasonality": "Trend + promo + season"}
        bar = go.Figure(go.Bar(x=[short[x["name"]] for x in F["models"]], y=m["WAPE %"],
                               text=m["WAPE %"].round(1).astype(str) + "%", textposition="outside",
                               marker_color=[TEAL if x["name"] == F["best"] else "#c9d1de" for x in F["models"]]))
        bar.update_yaxes(title="WAPE %", range=[0, m["WAPE %"].max() * 1.25])
        st.plotly_chart(style(bar, 230).update_layout(hovermode="closest", margin=dict(l=8, r=8, t=10, b=8)),
                        use_container_width=True)
        st.dataframe(m.round(2), hide_index=True, use_container_width=True)
    best = next(x for x in F["models"] if x["name"] == F["best"])
    st.markdown(f'<div class="note"><b>Result:</b> {F["best"]} has the lowest holdout WAPE '
                f'({best["wape"]:.1f}%, MAE {best["mae"]:.1f} units). The last {F["test_weeks"]} weeks are held out '
                f'chronologically (no shuffling), then the best model is refitted on all {F["n_weeks"]} weeks to forecast. '
                f'Its holdout error sets σD = {F["sigma"]:.1f}. Upload new data in the sidebar and everything refreshes.</div>',
                unsafe_allow_html=True)
    st.caption(f"Data: {st.session_state.source}. Synthetic data is simulated, not observed business data.")

# ---------- B. Safety stock and risk ----------
with tab2:
    st.markdown("### Analytical safety stock")
    st.latex(r"SS = Z\sqrt{\sigma_D^2 L + \bar D^2 \sigma_L^2}\qquad ROP = \bar D L + SS")
    st.markdown(f'<div class="note">Z = {SS["z"]:.3f} is the normal quantile for {service:.1%} cycle service. '
                f'SS = {SS["z"]:.3f} × √({sd:.1f}² × {L:g} + {d:.1f}² × {sL:g}²) = <b>{SS["ss"]:,.1f}</b> units, '
                f'so ROP = {d:.1f} × {L:g} + {SS["ss"]:,.1f} = <b>{SS["rop"]:,.1f}</b>. '
                f'Assumes demand and lead time are independent and roughly normal, in weeks and units per week.</div>',
                unsafe_allow_html=True)
    names = {0: ("Baseline", NAVY), 1: ("Supplier delay", AMBER), 2: ("Demand spike", RED)}
    h = go.Figure()
    for i, sc in enumerate(R["scenarios"]):
        h.add_trace(go.Bar(x=R["bins"], y=sc["counts"], name=names[i][0], marker_color=names[i][1],
                           opacity=0.62 if i else 0.85))
    h.add_vline(x=R["rop"], line=dict(color=TEAL, width=2.5), annotation_text=f"ROP {R['rop']:,.0f}",
                annotation_font_color=TEAL)
    h.update_layout(barmode="overlay", xaxis_title="Simulated lead-time demand (units)", yaxis_title="Runs")
    c1, c2 = st.columns([1.4, 1.2])
    with c1:
        st.markdown("### Monte Carlo lead-time demand")
        st.plotly_chart(style(h, 390).update_layout(hovermode="closest"), use_container_width=True)
    with c2:
        st.markdown("### Scenario risk")
        t = pd.DataFrame([{"Scenario": x["name"].split(" (")[0], "Stockout": f"{x['stockout']:.1%}",
                           "P95 demand": round(x["p95"]), "MC ROP": round(x["mc_rop"]),
                           "Formula ROP": round(x["formula_rop"])} for x in R["scenarios"]])
        st.dataframe(t, hide_index=True, use_container_width=True)
        st.markdown('<div class="note warn"><b>Reading it:</b> Stockout is the share of runs where lead-time demand '
                    'exceeds the baseline ROP. MC ROP is the simulated service-level quantile; Formula ROP is the '
                    'analytical answer for that scenario.</div>', unsafe_allow_html=True)
    st.caption(f"Simulation: {R['n']:,} runs, seed {inv.SEED}. Lead time ~ Normal(L, σL) floored at 0.1 week; weekly demand ~ "
               "Normal(D̄, σD). Scenarios: supplier delay adds 1 week and doubles σL; demand spike lifts demand 30% and σD 1.5x.")

# ---------- C. Reorder optimization ----------
with tab3:
    st.markdown("### Bayesian optimization of the reorder point")
    a = st.columns(5)
    unit = a[0].number_input("Unit cost", 1.0, 1e5, 20.0, 1.0)
    hold = a[1].number_input("Holding rate / yr", 0.01, 1.0, 0.25, 0.01)
    oc = a[2].number_input("Order cost", 0.0, 1e5, 100.0, 10.0)
    pen = a[3].number_input("Shortage cost / unit", 0.0, 1e5, 15.0, 1.0)
    trials = a[4].slider("Trials", 10, 100, 40, 5)
    st.markdown(f'<div class="note">Optuna TPE (sequential, model-based) picks the integer ROP that minimizes '
                f'<b>ordering + holding + expected shortage cost per year</b>, subject to the constraint '
                f'<b>cycle service level ≥ {service:.1%}</b>. Order quantity is fixed at EOQ. Costs are illustrative '
                f'assumptions, so the result is optimal for this model only, not for a real company.</div>', unsafe_allow_html=True)
    if st.button("Run optimization"):
        with st.spinner("Optimizing..."):
            st.session_state.opt = run_opt(d=d, sd=sd, L=L, sL=sL, service=service, unit_cost=unit, holding_rate=hold,
                                           order_cost=oc, shortage_cost=pen, n_trials=trials, n=n_sim)
            st.session_state.opt_sig = (d, sd, L, sL, service, unit, hold, oc, pen, trials, n_sim)
    O = st.session_state.get("opt")
    if O and st.session_state.opt_sig != (d, sd, L, sL, service, unit, hold, oc, pen, trials, n_sim):
        st.info("Inputs changed since the last run. Press Run optimization to refresh.")
    if O and O["best"]:
        b, bl = O["best"], O["baseline"]
        saving = bl["cost"] - b["cost"]
        m = st.columns(4)
        kpi(m[0], "Best feasible ROP", f"{b['rop']:,}", f"analytical baseline {bl['rop']:,.0f}", TEAL)
        kpi(m[1], "Expected annual cost", f"{b['cost']:,.0f}", f"baseline {bl['cost']:,.0f} ({saving:+,.0f} saved)")
        kpi(m[2], "Service level", f"{b['service']:.1%}", f"target {service:.1%}; baseline {bl['service']:.1%}")
        kpi(m[3], "Order quantity (EOQ)", f"{O['q']:,.0f}", "units per order")
        tr = pd.DataFrame(O["trials"])
        cv = go.Figure()
        for feas, col, nm in [(True, TEAL, "Feasible trial"), (False, RED, "Violates service level")]:
            sub = tr[tr.feasible == feas]
            cv.add_trace(go.Scatter(x=sub.trial, y=sub.cost, mode="markers", name=nm, marker=dict(color=col, size=8, opacity=.75),
                                    customdata=sub[["rop", "service"]], hovertemplate="ROP %{customdata[0]}<br>service %{customdata[1]:.1%}<br>cost %{y:,.0f}<extra></extra>"))
        cv.add_trace(go.Scatter(x=tr.trial, y=tr.best, name="Best feasible so far", line=dict(color=NAVY, width=3, shape="hv")))
        cv.update_layout(xaxis_title="Trial", yaxis_title="Annual cost")
        st.markdown("### Convergence")
        st.plotly_chart(style(cv, 360).update_layout(hovermode="closest"), use_container_width=True)
        st.markdown(f'<div class="note">Optuna searched ROP in [{O["range"][0]}, {O["range"][1]}] over {len(tr)} trials. '
                    f'{int(tr.feasible.sum())} met the service constraint. Best feasible ROP = {b["rop"]:,} '
                    f'with simulated service {b["service"]:.1%}. The analytical ROP reaches {bl["service"]:.1%} in the same '
                    f'simulation, so the formula is a good starting point but the simulation shows where it falls short or over-buffers.</div>',
                    unsafe_allow_html=True)
    elif O:
        st.warning("No feasible ROP found in the search range. Lower the service target or add trials.")

# ---------- D. Adaptive intelligence ----------
with tab4:
    st.markdown("### What happens when customer demand changes?")
    st.markdown('<div class="note"><b>Predictive + adaptive + real-time.</b> The last 26 weeks are replayed as if data arrived '
                'week by week. Every week the model <b>refits</b> (recent weeks weigh more), σD is <b>re-estimated</b> from its own '
                'recent forecast errors, and safety stock and ROP are <b>recomputed</b>. A static ROP, set once before the replay, '
                'is the comparison. A miss means the demand during lead time was higher than the ROP held that week. '
                'Use the demand-change slider in the sidebar to inject a change into the most recent weeks.</div>', unsafe_allow_html=True)
    P = run_replay(st.session_state.csv, L, sL, service, shock, shock_w)
    rows = pd.DataFrame(P["rows"])
    m = st.columns(4)
    kpi(m[0], "Misses: static ROP", f"{P['misses_static']} / {P['weeks']}", "weeks demand outran the fixed ROP", RED if P["misses_static"] else TEAL)
    kpi(m[1], "Misses: adaptive ROP", f"{P['misses_adaptive']} / {P['weeks']}", "same weeks, ROP recomputed weekly", RED if P["misses_adaptive"] else TEAL)
    kpi(m[2], "Latest adaptive ROP", f"{P['latest']['rop']:,.0f}", f"static ROP stays at {P['avg_rop_static']:,.0f}")
    kpi(m[3], "Latest σD (re-estimated)", f"{P['latest']['sd']:,.1f}", f"forecast demand {P['latest']['d']:,.0f} per week")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Forecast reacts to new demand")
        f1 = go.Figure()
        f1.add_trace(go.Scatter(mode="lines", x=rows.week, y=rows.original, name="Original demand", line=dict(color="#c9d1de", width=2, dash="dot")))
        f1.add_trace(go.Scatter(mode="lines", x=rows.week, y=rows.actual, name="Actual demand (with change)", line=dict(color=NAVY, width=2.5)))
        f1.add_trace(go.Scatter(mode="lines", x=rows.week, y=rows.forecast_next, name="Next-week forecast", line=dict(color=TEAL, width=2.5)))
        if P["shock_week"]:
            f1.add_shape(type="line", x0=P["shock_week"], x1=P["shock_week"], y0=0, y1=1, yref="paper", line=dict(color=AMBER, dash="dash", width=1.5))
            f1.add_annotation(x=P["shock_week"], y=1, yref="paper", text="demand change", showarrow=False, xanchor="left", font=dict(color=AMBER, size=11))
        f1.update_layout(yaxis_title="Units per week")
        st.plotly_chart(style(f1, 360), use_container_width=True)
    with c2:
        st.markdown("#### Reorder point: static vs adaptive")
        f2 = go.Figure()
        f2.add_trace(go.Scatter(mode="lines", x=rows.week, y=rows.rop_static, name="Static ROP", line=dict(color=AMBER, width=2.5, dash="dash")))
        f2.add_trace(go.Scatter(mode="lines", x=rows.week, y=rows.rop_adaptive, name="Adaptive ROP", line=dict(color=TEAL, width=3)))
        f2.add_trace(go.Scatter(mode="markers", x=rows.week, y=rows.realized_ltd, name="Realized lead-time demand",
                                marker=dict(color=NAVY, size=6, opacity=.7)))
        miss = rows[rows.miss_static]
        f2.add_trace(go.Scatter(mode="markers", x=miss.week, y=miss.realized_ltd, name="Missed by static ROP",
                                marker=dict(color=RED, size=11, symbol="x")))
        f2.update_layout(yaxis_title="Units")
        st.plotly_chart(style(f2, 360), use_container_width=True)
    O2 = run_opt(d=P["latest"]["d"], sd=P["latest"]["sd"], L=L, sL=sL, service=service, unit_cost=unit, holding_rate=hold,
                 order_cost=oc, shortage_cost=pen, n_trials=25, n=n_sim)
    if O2["best"]:
        st.markdown(f'<div class="note"><b>Real-time re-optimization.</b> Using the latest forecast ({P["latest"]["d"]:,.0f}/week) and re-estimated '
                    f'σD ({P["latest"]["sd"]:,.1f}), Optuna TPE re-ran 25 trials on the spot: best feasible ROP '
                    f'<b>{O2["best"]["rop"]:,}</b> at {O2["best"]["service"]:.1%} service, expected annual cost {O2["best"]["cost"]:,.0f}. '
                    f'The formula alone gives {P["latest"]["rop"]:,.0f}. Move the sidebar demand slider and this answer updates immediately.</div>', unsafe_allow_html=True)
    with st.expander("How the prediction works"):
        st.markdown("Demand is predicted with a regression on the calendar and promotions:")
        st.latex(r"\hat D_t = a + b\,t + c\,\text{promo}_t + d\sin\tfrac{2\pi\,doy}{365} + e\cos\tfrac{2\pi\,doy}{365}")
        st.markdown("Each week it is refitted with weights 0.95 per week of age, so a lasting change in demand shifts the fit within a few "
                    "weeks while old history still sets the seasonal shape. The forecast error of the last 12 weeks becomes σD, so a surprise "
                    "automatically widens safety stock.")

# ---------- E. AI sensitivity ----------
with tab5:
    st.markdown("### AI-driven sensitivity analysis")
    st.markdown('<div class="note">A manual sweep tries every input combination on a grid. Here Optuna TPE <b>searches the uncertainty '
                'range intelligently</b>: it hunts for the input combination that hurts a fixed ROP the most (highest simulated stockout '
                'probability), then ranks which inputs matter most. Ranges tested: demand 0.8x to 1.4x, demand variability 0.7x to 2.0x, '
                'lead time 0.75x to 1.5x, lead-time variability 0.5x to 2.5x.</div>', unsafe_allow_html=True)
    opts = {"Analytical ROP": SS["rop"]}
    if st.session_state.get("opt") and st.session_state.opt["best"]:
        opts["Optimized ROP (tab 3)"] = float(st.session_state.opt["best"]["rop"])
    z1, z2, z3 = st.columns([1.2, 1, 1])
    which = z1.selectbox("ROP to stress-test", list(opts))
    n_s = z2.slider("Search trials", 30, 120, 60, 10)
    z3.write("")
    go_s = z3.button("Run AI sensitivity")
    key = (d, sd, L, sL, service, opts[which], n_s)
    if go_s:
        with st.spinner("Searching for worst cases..."):
            st.session_state.sens, st.session_state.sens_key = run_sens(*key), key
    S = st.session_state.get("sens")
    if S and st.session_state.sens_key != key:
        st.info("Inputs changed since the last run. Press Run AI sensitivity to refresh.")
    if S:
        w = S["worst"]
        m = st.columns(4)
        kpi(m[0], "Tested ROP", f"{S['rop']:,.0f}", "stress-tested value")
        kpi(m[1], "Nominal stockout risk", f"{S['nominal']:.1%}", "with today's assumptions", TEAL)
        kpi(m[2], "Worst case found", f"{w['stockout']:.0%}", f"after {S['n_trials']} trials", RED)
        kpi(m[3], "ROP needed in worst case", f"{w['rop_needed']:,.0f}", f"to still hold {service:.1%} service")
        st.write("")
        lab = {"demand_mult": "Demand level", "variability_mult": "Demand variability",
               "lead_time_mult": "Lead time", "lead_time_var_mult": "Lead-time variability"}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Which inputs matter most")
            imp = pd.Series(S["importance"]).sort_values()
            fi = go.Figure(go.Bar(x=imp.values, y=[lab[i] for i in imp.index], orientation="h",
                                  marker_color=[TEAL if v == imp.max() else "#c9d1de" for v in imp.values],
                                  text=[f"{v:.0%}" for v in imp.values], textposition="outside"))
            fi.update_xaxes(range=[0, imp.max() * 1.25], tickformat=".0%")
            st.plotly_chart(style(fi, 330).update_layout(hovermode="closest"), use_container_width=True)
        with c2:
            st.markdown("#### Search progress")
            tr = pd.DataFrame(S["trials"])
            fs = go.Figure()
            fs.add_trace(go.Scatter(x=tr.trial, y=tr.stockout, mode="markers", name="Scenario tried", marker=dict(color=BLUE, size=7, opacity=.6)))
            fs.add_trace(go.Scatter(x=tr.trial, y=tr.best, mode="lines", name="Worst case so far", line=dict(color=RED, width=3, shape="hv")))
            fs.update_layout(xaxis_title="Trial", yaxis_title="Stockout probability", yaxis_tickformat=".0%")
            st.plotly_chart(style(fs, 330).update_layout(hovermode="closest"), use_container_width=True)
        t = pd.DataFrame({"Input": list(lab.values()), "Nominal": ["1.00x"] * 4,
                          "Worst case found": [f"{w[k]:.2f}x" for k in inv.SENS_RANGES]})
        st.dataframe(t, hide_index=True, use_container_width=True)
        top = max(S["importance"], key=S["importance"].get)
        st.markdown(f'<div class="note warn"><b>Reading it:</b> {lab[top]} has the largest influence on stockout risk. A full grid with 5 '
                    f'values per input would need {S["grid_cost"]} simulations; the Bayesian search used {S["n_trials"]}. '
                    f'Importance shows influence inside the tested ranges, so a wider range can change the ranking.</div>', unsafe_allow_html=True)

# ---------- B. Before AI vs With AI ----------
with tabB:
    st.markdown("### Before AI vs With AI")
    what = f"a {shock:+.0%} demand change in the last {shock_w} weeks" if shock else "stable demand"
    st.markdown(f'<div class="note"><b>Same data, same 26-week backtest, same what-if ({what}).</b> '
                '<b>Before AI</b> is the classic reorder-point calculator: average of all history, its raw standard deviation, '
                'ROP = D̄L + Zσ√L, set once and left alone. It ignores lead-time variability and trend. '
                '<b>With AI</b> is this system: machine-learning forecast, re-estimated risk, Monte Carlo, weekly updates and Bayesian optimization. '
                'The Before AI case is a modelled baseline, not a real company record.</div>', unsafe_allow_html=True)
    cl = P["classic"]
    ai = O2["best"]
    cost_b = inv.policy_eval(P["rop_classic"], P["latest"]["d"], P["latest"]["sd"], L, sL, n=n_sim,
                             unit_cost=unit, holding_rate=hold, order_cost=oc, shortage_cost=pen)
    cov_b, cov_a = 1 - P["misses_classic"] / P["weeks"], 1 - P["misses_adaptive"] / P["weeks"]
    k = st.columns(4)
    kpi(k[0], "Weeks demand covered", f"{cov_a:.0%}", f"Before AI: {cov_b:.0%} | target {service:.0%}", TEAL)
    kpi(k[1], "Units short in 26 weeks", f"{P['short_adaptive']:,.0f}", f"Before AI: {P['short_classic']:,.0f}", TEAL)
    kpi(k[2], "Next-week forecast error", f"{P['wape_ai']:.1f}%", f"Before AI (average): {P['wape_avg']:.1f}% WAPE", TEAL)
    if ai:
        kpi(k[3], "Expected annual cost", f"{ai['cost']:,.0f}", f"Before AI: {cost_b['cost']:,.0f} at {cost_b['service']:.0%} service", TEAL)
    st.write("")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("#### Reorder point over time")
        fb = go.Figure()
        fb.add_trace(go.Scatter(mode="lines", x=rows.week, y=rows.rop_classic, name="Before AI: classic ROP", line=dict(color=AMBER, width=2.5, dash="dash")))
        fb.add_trace(go.Scatter(mode="lines", x=rows.week, y=rows.rop_adaptive, name="With AI: adaptive ROP", line=dict(color=TEAL, width=3)))
        fb.add_trace(go.Scatter(mode="markers", x=rows.week, y=rows.realized_ltd, name="Realized lead-time demand", marker=dict(color=NAVY, size=6, opacity=.7)))
        mc = rows[rows.miss_classic]
        fb.add_trace(go.Scatter(mode="markers", x=mc.week, y=mc.realized_ltd, name="Missed by Before AI", marker=dict(color=RED, size=11, symbol="x")))
        fb.update_layout(yaxis_title="Units")
        st.plotly_chart(style(fb, 400), use_container_width=True)
    with c2:
        st.markdown("#### How the approaches differ")
        cmp_df = pd.DataFrame({
            "Step": ["Demand estimate", "Variability (σD)", "Lead-time risk", "Updating", "Risk view", "Reorder point", "Demand change"],
            "Before AI": [f"{cl['d']:.0f}/wk, average of all history", f"{cl['sd']:.1f}, raw spread of history", "Ignored",
                          "Manual, rarely", "None", f"{P['rop_classic']:.0f}, fixed formula", "Not noticed"],
            "With AI": [f"{P['latest']['d']:.0f}/wk, ML forecast", f"{P['latest']['sd']:.1f}, recent forecast error", "In formula and Monte Carlo",
                        "Automatic, weekly", "Monte Carlo + AI sensitivity", f"{O2['best']['rop'] if ai else P['latest']['rop']:.0f}, Bayesian optimized",
                        "Detected, ROP and σD adapt"]})
        st.dataframe(cmp_df, hide_index=True, use_container_width=True, height=300)
    extra = P["avg_rop_adaptive"] - P["rop_classic"]
    st.markdown(f'<div class="note warn"><b>Honest reading:</b> the AI policy carries on average {extra:+,.0f} units of reorder point versus the classic one, '
                f'and in return cuts shortages from {P["short_classic"]:,.0f} to {P["short_adaptive"]:,.0f} units. '
                f'When demand is stable, a plain average forecasts almost as well as the model ({P["wape_avg"]:.1f}% vs {P["wape_ai"]:.1f}% error); '
                'the clear AI advantage appears when demand changes, because it notices and adapts. Move the sidebar slider to test both cases.</div>',
                unsafe_allow_html=True)

# ---------- F. Autopilot ----------
with tabA:
    st.markdown("### Autopilot: automated reorder decision")
    st.markdown('<div class="note">A rule-based decision agent sits on top of the models. Each week it reads the refreshed forecast, risk and '
                'ROP, compares them with your inventory position, and recommends an action in plain language. It also watches forecast errors '
                'for drift and anomalies and triggers re-optimization on its own. Enter your current stock to see the decision.</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    on_hand = a1.number_input("Inventory on hand (units)", 0, 100000, 450, 10)
    on_order = a2.number_input("Already on order (units)", 0, 100000, 0, 10)
    D = inv.autopilot(P, on_hand, on_order, O2["q"], R["scenarios"][1]["stockout"], service)
    col = RED if D["reorder"] else TEAL
    head = f"REORDER NOW: order {D['qty']:,} units" if D["reorder"] else "HOLD: no order needed this week"
    st.markdown(f'<div class="kpi" style="border-left:5px solid {col}"><div class="l">Recommended action</div>'
                f'<div class="v" style="color:{col}">{head}</div>'
                f'<div class="n">Inventory position {D["position"]:,.0f} vs reorder point {D["rop"]:,.0f} | '
                f'{D["cover_weeks"]:.1f} weeks of cover at forecast demand | order size = EOQ batches of {O2["q"]:,.0f}</div></div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("#### What the AI did automatically")
        r0, r1 = rows.iloc[0], rows.iloc[-1]
        trig = f"Triggered: drift detected. Optuna re-ran 25 trials, best feasible ROP {O2['best']['rop']:,}." if (D["retrain"] and O2["best"]) \
            else "Not needed: demand is tracking the forecast."
        steps = [f"Refitted the forecast on {F['n_weeks']} weeks: next week {r1.forecast_next:,.0f} units.",
                 f"Re-estimated σD from recent errors: {r0.sd:,.1f} to {r1.sd:,.1f}.",
                 f"Recomputed safety stock and ROP: {D['rop_before']:,.0f} to {D['rop']:,.0f}.",
                 f"Bayesian re-optimization. {trig}"]
        st.markdown("".join(f'<div class="note" style="margin:.3rem 0">{i + 1}. {t}</div>' for i, t in enumerate(steps)), unsafe_allow_html=True)
        st.markdown("#### Alerts")
        colors = {"red": RED, "amber": AMBER, "green": TEAL}
        st.markdown("".join(f'<div class="note" style="border-left-color:{colors[l]};margin:.3rem 0">{t}</div>' for l, t in D["alerts"]),
                    unsafe_allow_html=True)
    with c2:
        st.markdown("#### Demand monitor: standardized forecast error")
        zc = [RED if abs(v) > 2.5 else (AMBER if abs(v) > 1.5 else "#9fb0c8") for v in rows.z]
        fz = go.Figure(go.Bar(x=rows.week, y=rows.z, marker_color=zc, name="Forecast error (σ units)"))
        for yv in (2.5, -2.5):
            fz.add_shape(type="line", xref="paper", x0=0, x1=1, y0=yv, y1=yv, line=dict(color=RED, dash="dash", width=1))
        fz.update_layout(yaxis_title="Standard deviations", showlegend=False)
        st.plotly_chart(style(fz, 380).update_layout(hovermode="closest"), use_container_width=True)
        st.caption("Bars beyond the dashed lines are anomalies. Several consecutive bars on one side signal drift, which raises or lowers the ROP automatically.")

st.markdown('<div class="foot">Atharva Soundankar | as25dxb020 | MAIB | Batch September 2025 | SP Jain School of Global Management | '
            'AI in Operations, Session II: AI Applications in Reorder Point and Inventory Management</div>', unsafe_allow_html=True)
