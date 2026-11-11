"""Core calculations for the Session II inventory exercise.

Atharva Soundankar | as25dxb020 | MAIB | AI in Operations
All demand quantities are in units per WEEK and all lead times are in WEEKS.
"""
import numpy as np
import pandas as pd
import optuna
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------- Data ----------
def make_synthetic(weeks=104, seed=SEED):
    """Clearly labeled SYNTHETIC weekly demand for one SKU: trend + annual season + promotions + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-11-11", periods=weeks, freq="W-MON")
    t = np.arange(weeks)
    angle = 2 * np.pi * dates.dayofyear / 365.25
    promo = (rng.random(weeks) < 0.12).astype(int)
    level = 120 + 0.25 * t + 18 * np.sin(angle) + 8 * np.cos(angle)
    demand = rng.poisson(level * (1 + 0.25 * promo)) + rng.normal(0, 6, weeks)
    return pd.DataFrame({"week": dates.strftime("%Y-%m-%d"), "sku": "SKU-A101",
                         "demand": np.maximum(demand, 0).round().astype(int), "promo": promo})


def validate(df):
    """Check the upload schema: week (date), demand (>= 0), optional promo (0/1) and sku."""
    df = df.rename(columns=str.strip).rename(columns=str.lower)
    missing = {"week", "demand"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    if df[["week", "demand"]].isna().any().any():
        raise ValueError("Some rows have an invalid date or a missing/non-numeric demand value")
    if (df["demand"] < 0).any():
        raise ValueError("Demand values must be zero or positive")
    df["promo"] = pd.to_numeric(df.get("promo", 0), errors="coerce").fillna(0).clip(0, 1).astype(int)
    df["sku"] = df.get("sku", "SKU")
    df = df.sort_values("week").drop_duplicates("week", keep="last").reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("At least 30 weeks of history are needed")
    df["week"] = df["week"].dt.strftime("%Y-%m-%d")
    return df[["week", "sku", "demand", "promo"]]


# ---------- 1. Demand forecasting ----------
def _features(dates, t, promo, seasonal):
    cols = [t, promo]
    if seasonal:
        a = 2 * np.pi * pd.DatetimeIndex(dates).dayofyear / 365.25
        cols += [np.sin(a), np.cos(a)]
    return np.column_stack(cols)


def _metrics(y, p):
    err = np.abs(y - p)
    return {"mae": float(err.mean()), "wape": float(err.sum() / y.sum() * 100)}


def forecast(df, horizon=8, test_weeks=12):
    """Chronological split: train on all but the last `test_weeks`, compare models on the holdout,
    then refit the best one on the full history and forecast `horizon` weeks (promo assumed 0)."""
    y = df["demand"].to_numpy(float)
    dates = pd.to_datetime(df["week"])
    t = np.arange(len(y), dtype=float)
    promo = df["promo"].to_numpy(float)
    split = len(y) - test_weeks
    tr, te = slice(0, split), slice(split, None)

    preds = {"4-week moving average (baseline)": np.full(test_weeks, y[split - 4:split].mean())}
    for name, seasonal in [("Linear regression: trend + promo", False),
                           ("Linear regression: trend + promo + seasonality", True)]:
        m = LinearRegression().fit(_features(dates[tr], t[tr], promo[tr], seasonal), y[tr])
        preds[name] = m.predict(_features(dates[te], t[te], promo[te], seasonal))
    models = [{"name": k, **_metrics(y[te], v)} for k, v in preds.items()]
    best = min(models, key=lambda m: m["wape"])["name"]
    sigma = float(np.sqrt(np.mean((y[te] - preds[best]) ** 2)))  # holdout RMSE = forecast error sd

    fdates = dates.iloc[-1] + pd.to_timedelta(np.arange(1, horizon + 1) * 7, unit="D")
    ft = np.arange(len(y), len(y) + horizon, dtype=float)
    if best.startswith("4-week"):
        fc = np.full(horizon, y[-4:].mean())
    else:
        seasonal = "seasonality" in best
        m = LinearRegression().fit(_features(dates, t, promo, seasonal), y)
        fc = m.predict(_features(fdates, ft, np.zeros(horizon), seasonal))
    fc = np.maximum(fc, 0)

    series = [{"week": w, "actual": float(a)} for w, a in zip(df["week"], y)]
    for i, row in enumerate(series[split:]):
        row["baseline"] = float(preds["4-week moving average (baseline)"][i])
        row["model"] = float(preds[best][i])
    series[-1]["forecast"] = series[-1]["actual"]  # joins the forecast line to history
    series += [{"week": d.strftime("%Y-%m-%d"), "forecast": float(v),
                "band": [float(max(v - 1.96 * sigma, 0)), float(v + 1.96 * sigma)]}
               for d, v in zip(fdates, fc)]
    return {"series": series, "models": models, "best": best, "sigma": sigma,
            "mean_forecast": float(fc.mean()), "split_week": df["week"].iloc[split],
            "n_weeks": len(y), "test_weeks": test_weeks, "horizon": horizon}


# ---------- 2. Dynamic safety stock ----------
def safety_stock(d, sd, L, sL, service):
    """SS = Z * sqrt(sigma_D^2 * L + D_bar^2 * sigma_L^2),  ROP = D_bar * L + SS."""
    z = float(norm.ppf(service))
    ss = z * np.sqrt(sd ** 2 * L + d ** 2 * sL ** 2)
    return {"z": z, "ss": float(ss), "rop": float(d * L + ss)}


def simulate_ltd(d, sd, L, sL, n=2000, seed=SEED):
    """Monte Carlo lead-time demand: lead time ~ Normal(L, sL) floored at 0.1 week,
    weekly demand ~ Normal(d, sd) so demand over lead time ~ Normal(d*LT, sd^2*LT), floored at 0."""
    rng = np.random.default_rng(seed)
    lt = np.maximum(rng.normal(L, sL, n), 0.1)
    return np.maximum(rng.normal(d * lt, sd * np.sqrt(lt)), 0)


SCENARIOS = {  # name: (demand multiplier, demand sd multiplier, extra lead time weeks, lead time sd multiplier)
    "Baseline": (1.0, 1.0, 0.0, 1.0),
    "Supplier delay (+1 week, 2x lead-time variability)": (1.0, 1.0, 1.0, 2.0),
    "Demand spike (+30% demand, 1.5x variability)": (1.3, 1.5, 0.0, 1.0),
}


def risk_scenarios(d, sd, L, sL, service, n=2000, bins=30):
    """Stockout risk of the analytical (baseline) ROP under each scenario, with a shared histogram."""
    rop = safety_stock(d, sd, L, sL, service)["rop"]
    sims = {k: simulate_ltd(d * a, sd * b, L + c, sL * e, n) for k, (a, b, c, e) in SCENARIOS.items()}
    edges = np.linspace(0, max(s.max() for s in sims.values()), bins + 1)
    rows = []
    for k, s in sims.items():
        a, b, c, e = SCENARIOS[k]
        rows.append({"name": k, "mean": float(s.mean()), "p50": float(np.percentile(s, 50)),
                     "p95": float(np.percentile(s, 95)), "stockout": float((s > rop).mean()),
                     "mc_rop": float(np.quantile(s, service)),
                     "formula_rop": safety_stock(d * a, sd * b, L + c, sL * e, service)["rop"],
                     "counts": np.histogram(s, edges)[0].tolist()})
    mids = ((edges[:-1] + edges[1:]) / 2).round(0).tolist()
    return {"rop": rop, "scenarios": rows, "bins": mids, "n": n}


# ---------- 3. Bayesian optimization of the reorder point ----------
def _cost(ltd, d, rop, unit_cost=20.0, holding_rate=0.25, order_cost=100.0, shortage_cost=15.0):
    """Illustrative annual cost of a reorder point: ordering + holding + expected shortage, EOQ order quantity.
    Returns (cost, cycle service level, EOQ)."""
    annual = 52 * d
    h = unit_cost * holding_rate
    q = float(np.sqrt(2 * annual * order_cost / h))
    cycles = annual / q
    short = np.maximum(ltd - rop, 0).mean()
    leftover = np.maximum(rop - ltd, 0).mean()
    cost = order_cost * cycles + h * (q / 2 + leftover) + shortage_cost * short * cycles
    return float(cost), float((ltd <= rop).mean()), q


def policy_eval(rop, d, sd, L, sL, n=2000, seed=SEED, **costs):
    """Cost and service of ANY reorder point under the same simulation and cost model as the optimizer."""
    cost, svc, _ = _cost(simulate_ltd(d, sd, L, sL, n, seed), d, rop, **costs)
    return {"cost": cost, "service": svc}


def optimize_rop(d, sd, L, sL, service, unit_cost=20.0, holding_rate=0.25, order_cost=100.0,
                 shortage_cost=15.0, n_trials=40, n=2000, seed=SEED):
    """Optuna TPE chooses the ROP minimizing illustrative annual cost
    (ordering + holding + expected shortage) subject to cycle service level >= target.
    Order quantity is fixed at EOQ so ROP is the single, non-redundant decision."""
    ltd = simulate_ltd(d, sd, L, sL, n, seed)  # common random numbers for every trial
    costs = dict(unit_cost=unit_cost, holding_rate=holding_rate, order_cost=order_cost, shortage_cost=shortage_cost)
    q = _cost(ltd, d, d * L, **costs)[2]

    def evaluate(rop):
        return _cost(ltd, d, rop, **costs)[:2]

    lo, hi = int(d * L * 0.8), int(np.quantile(ltd, 0.999) * 1.15) + 1

    def objective(trial):
        cost, svc = evaluate(trial.suggest_int("rop", lo, hi))
        trial.set_user_attr("service", svc)
        trial.set_user_attr("gap", service - svc)
        trial.set_constraint("service", service - svc)  # feasible when <= 0
        return cost

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    trials, best, best_so_far = [], None, None
    for t in study.trials:
        feasible = t.user_attrs["gap"] <= 0
        if feasible and (best is None or t.value < best.value):
            best = t
        best_so_far = best.value if best else None
        trials.append({"trial": t.number + 1, "rop": t.params["rop"], "cost": t.value,
                       "service": t.user_attrs["service"], "feasible": feasible, "best": best_so_far})
    base_rop = safety_stock(d, sd, L, sL, service)["rop"]
    base_cost, base_svc = evaluate(base_rop)
    return {"trials": trials, "q": q, "range": [lo, hi],
            "best": {"rop": best.params["rop"], "cost": best.value, "service": best.user_attrs["service"]}
            if best else None,
            "baseline": {"rop": base_rop, "cost": base_cost, "service": base_svc}}


# ---------- 4. Adaptive intelligence: rolling refit, adaptive ROP, demand shocks ----------
def adaptive_replay(df, L, sL, service, shock_pct=0.0, shock_weeks=10, replay_weeks=26, decay=0.95, err_window=12):
    """Replay the last `replay_weeks` as if data arrived week by week.
    Each week the recency-weighted seasonal regression is refitted, sigma_D is re-estimated from its recent
    one-step forecast errors, and SS/ROP are recomputed. A static ROP (set once, before the replay) is the comparison.
    `shock_pct` multiplies demand in the final `shock_weeks` (a what-if demand change).
    A miss = realized lead-time demand exceeded the ROP held that week."""
    y = df["demand"].to_numpy(float).copy()
    n = len(y)
    shock_weeks = min(shock_weeks, replay_weeks - 2)
    if shock_pct:
        y[n - shock_weeks:] *= 1 + shock_pct
    dates = pd.to_datetime(df["week"])
    t = np.arange(n, dtype=float)
    X = _features(dates, t, df["promo"].to_numpy(float), True)
    Lw = max(1, int(round(L)))
    first, last = n - replay_weeks - Lw, n - 1 - Lw  # origins whose realized lead-time demand is known

    def fit_forecast(u):  # fit on weeks 0..u (recent weeks weigh more), forecast the next Lw weeks, promo assumed 0
        m = LinearRegression().fit(X[:u + 1], y[:u + 1], sample_weight=decay ** (u - t[:u + 1]))
        fd = dates.iloc[u] + pd.to_timedelta(np.arange(1, Lw + 1) * 7, unit="D")
        return np.maximum(m.predict(_features(fd, t[u] + np.arange(1, Lw + 1), np.zeros(Lw), True)), 1)

    fc = {u: fit_forecast(u) for u in range(first - err_window, last + 1)}
    rows = []
    for e in range(first, last + 1):
        errs = np.array([y[u + 1] - fc[u][0] for u in range(e - err_window, e)])
        d_e, sd_e = float(fc[e].mean()), float(np.sqrt((errs ** 2).mean()))
        rop_a = safety_stock(d_e, sd_e, L, sL, service)["rop"]
        rows.append({"week": dates.iloc[e].strftime("%Y-%m-%d"), "actual": float(y[e]),
                     "original": float(df["demand"].iloc[e]), "forecast_next": float(fc[e][0]), "d": d_e, "sd": sd_e,
                     "rop_adaptive": float(rop_a), "realized_ltd": float(y[e + 1:e + Lw + 1].sum())})
    hist = y[:first + 1]  # BEFORE AI: textbook calculator on the average and std dev of all history, no lead-time variability
    rop_classic = float(hist.mean() * L + norm.ppf(service) * hist.std(ddof=1) * np.sqrt(L))
    for i, r in enumerate(rows):
        e = first + i
        r["rop_static"] = rows[0]["rop_adaptive"]
        r["rop_classic"] = rop_classic
        r["next_actual"] = float(y[e + 1])
        r["hist_avg"] = float(y[:e + 1].mean())
        prev_sd = rows[i - 1]["sd"] if i else r["sd"]
        r["z"] = float((y[e] - fc[e - 1][0]) / max(prev_sd, 1e-9))  # standardized forecast error (monitoring signal)
        r["miss_adaptive"] = bool(r["realized_ltd"] > r["rop_adaptive"])
        r["miss_static"] = bool(r["realized_ltd"] > r["rop_static"])
        r["miss_classic"] = bool(r["realized_ltd"] > rop_classic)
    nxt = np.array([r["next_actual"] for r in rows])
    z = np.array([r["z"] for r in rows])
    return {"rows": rows, "shock_week": dates.iloc[n - shock_weeks].strftime("%Y-%m-%d") if shock_pct else None,
            "misses_adaptive": sum(r["miss_adaptive"] for r in rows), "misses_static": sum(r["miss_static"] for r in rows),
            "avg_rop_adaptive": float(np.mean([r["rop_adaptive"] for r in rows])), "avg_rop_static": rows[0]["rop_adaptive"],
            "latest": {"d": rows[-1]["d"], "sd": rows[-1]["sd"], "rop": rows[-1]["rop_adaptive"]}, "weeks": len(rows),
            "misses_classic": sum(r["miss_classic"] for r in rows), "rop_classic": rop_classic,
            "classic": {"d": float(hist.mean()), "sd": float(hist.std(ddof=1))},
            "short_adaptive": float(sum(max(r["realized_ltd"] - r["rop_adaptive"], 0) for r in rows)),
            "short_classic": float(sum(max(r["realized_ltd"] - rop_classic, 0) for r in rows)),
            "wape_ai": float(np.abs(nxt - [r["forecast_next"] for r in rows]).sum() / nxt.sum() * 100),
            "wape_avg": float(np.abs(nxt - [r["hist_avg"] for r in rows]).sum() / nxt.sum() * 100),
            "drift": float(max((z[-12:][i - 3:i + 1].mean() for i in range(3, len(z[-12:]))), key=abs)), "anomalies": [rows[i]["week"] for i in np.where(np.abs(z) > 2.5)[0]]}


# ---------- 5. AI-driven sensitivity analysis (Bayesian search instead of a manual sweep) ----------
SENS_RANGES = {"demand_mult": (0.8, 1.4), "variability_mult": (0.7, 2.0),
               "lead_time_mult": (0.75, 1.5), "lead_time_var_mult": (0.5, 2.5)}


def ai_sensitivity(d, sd, L, sL, service, rop, n_trials=60, n=1000, seed=SEED):
    """Optuna TPE hunts the uncertainty box for the inputs that hurt a FIXED reorder point most
    (maximizes simulated stockout probability), then ranks which inputs matter via parameter importance."""
    def stockout(a, b, c, e):
        return float((simulate_ltd(d * a, sd * b, L * c, sL * e, n, seed) > rop).mean())

    def objective(trial):
        a, b, c, e = (trial.suggest_float(k, *r) for k, r in SENS_RANGES.items())
        p = stockout(a, b, c, e)
        trial.set_user_attr("rop_needed", safety_stock(d * a, sd * b, L * c, sL * e, service)["rop"])
        return p

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=20))
    study.optimize(objective, n_trials=n_trials)
    imp = optuna.importance.get_param_importances(study)
    best, run = study.best_trial, -1.0
    trials = []
    for tr in study.trials:
        run = max(run, tr.value)
        trials.append({"trial": tr.number + 1, "stockout": tr.value, "best": run, **tr.params})
    return {"importance": imp, "trials": trials, "rop": float(rop), "nominal": stockout(1, 1, 1, 1),
            "worst": {"stockout": best.value, "rop_needed": best.user_attrs["rop_needed"], **best.params},
            "grid_cost": 5 ** len(SENS_RANGES), "n_trials": n_trials}


# ---------- 6. Autopilot: automated reorder decision, drift alerts, automatic re-optimization ----------
def autopilot(P, on_hand, on_order, q, delay_risk=0.0, service=0.95):
    """Rule-based decision agent on top of the models. Orders whole EOQ batches until the inventory
    position (on hand + on order) clears the latest adaptive ROP; raises plain-language alerts."""
    rop, first_rop = P["latest"]["rop"], P["rows"][0]["rop_adaptive"]
    pos = on_hand + on_order
    qty = 0 if pos > rop else int(np.ceil((rop - pos + 1) / q) * q)
    shift = rop / first_rop - 1
    drift = P["drift"]  # strongest 4-week average standardized forecast error in the last 12 weeks
    alerts = []
    if drift > 1.5:
        alerts.append(("red", f"Demand drift upward: over the last 12 weeks, demand ran {drift:.1f} standard deviations above forecast "
                              f"for 4 weeks in a row. ROP moved {shift:+.0%} and the optimizer re-ran automatically."))
    elif drift < -1.5:
        alerts.append(("amber", f"Demand drift downward: demand ran {abs(drift):.1f} standard deviations below forecast for 4 weeks. "
                                f"ROP moved {shift:+.0%} to release working capital."))
    elif abs(shift) > 0.2:
        alerts.append(("amber", f"Demand level shifted: ROP moved {shift:+.0%} since the start of the replay, so the optimizer re-ran automatically."))
    else:
        alerts.append(("green", "Demand is tracking the forecast. No drift detected."))
    if P["anomalies"]:
        alerts.append(("amber", f"{len(P['anomalies'])} demand anomaly week(s) beyond 2.5 standard deviations: {', '.join(P['anomalies'][-3:])}."))
    if delay_risk > 1 - service + 0.15:
        alerts.append(("red", f"Supplier-delay risk: a one-week delay would raise stockout probability to {delay_risk:.0%}. Consider a backup supplier or a higher buffer."))
    return {"reorder": qty > 0, "qty": qty, "position": float(pos), "rop": float(rop), "rop_before": float(first_rop),
            "cover_weeks": float(pos / max(P["latest"]["d"], 1e-9)), "alerts": alerts,
            "retrain": abs(drift) > 1.5 or abs(shift) > 0.2}


if __name__ == "__main__":
    from pathlib import Path
    df = make_synthetic()
    out = Path(__file__).resolve().parent / "data" / "weekly_demand.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    # self-checks
    assert validate(pd.read_csv(out)).equals(df)
    r = safety_stock(100, 20, 2, 0.5, 0.95)
    assert abs(r["ss"] - 1.6448536 * np.sqrt(400 * 2 + 10000 * 0.25)) < 1e-4
    assert abs(r["rop"] - (200 + r["ss"])) < 1e-9
    assert np.array_equal(simulate_ltd(100, 20, 2, 0.5), simulate_ltd(100, 20, 2, 0.5))
    f = forecast(df)
    o = optimize_rop(f["mean_forecast"], f["sigma"], 2, 0.5, 0.95)
    assert o["best"]["service"] >= 0.95
    print("best model:", f["best"], "| sigma:", round(f["sigma"], 2), "| mean fc:", round(f["mean_forecast"], 1))
    print("optimized:", o["best"], "| baseline:", o["baseline"])
    ad = adaptive_replay(df, 2, 0.5, 0.95, shock_pct=0.4)
    assert ad["misses_adaptive"] <= ad["misses_static"]
    sens = ai_sensitivity(f["mean_forecast"], f["sigma"], 2, 0.5, 0.95, o["baseline"]["rop"], n_trials=30)
    assert sens["worst"]["stockout"] >= sens["nominal"]
    assert abs(policy_eval(o["best"]["rop"], f["mean_forecast"], f["sigma"], 2, 0.5)["cost"] - o["best"]["cost"]) < 1e-6
    ap = autopilot(ad, on_hand=300, on_order=0, q=o["q"])
    assert ap["reorder"] and ap["qty"] % round(o["q"]) < 1
    print("classic misses", ad["misses_classic"], "| wape ai/avg", round(ad["wape_ai"], 1), round(ad["wape_avg"], 1))
    print("adaptive misses", ad["misses_adaptive"], "static misses", ad["misses_static"])
    print("all checks passed")
