# scripts/scm_core.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def _simplex_project(v):
    """Euclidean projection of v onto the probability simplex (nonneg, sums to 1)."""
    v = np.asarray(v, dtype=float)
    if v.sum() == 1 and np.all(v >= 0):
        return v
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w

def fit_scm(y_df: pd.DataFrame, treated: str, donors: list, pre_years: list, ridge=1e-8):
    assert treated in y_df.columns, f"{treated} not in columns"
    donors = [d for d in donors if d in y_df.columns and d != treated]
    YT = y_df.loc[pre_years, treated].to_numpy()
    XD = y_df.loc[pre_years, donors].to_numpy()  # T x N
    N = XD.shape[1]
    if N == 0:
        raise ValueError("No donors available after filtering.")

    # Objective: minimize ||YT - XD @ w||^2 + ridge * ||w||^2
    def obj(w):
        diff = YT - XD.dot(w)
        return float(np.dot(diff, diff) + ridge * np.dot(w, w))

    # Constraints: w >= 0, sum w = 1
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w},  # elementwise >= 0
    )
    w0 = np.ones(N) / N
    res = minimize(obj, w0, constraints=cons, bounds=[(0,1)]*N, method="SLSQP", options={"maxiter": 10000})

    if not res.success:
        # Fall back to projected gradient descent on simplex
        w = w0.copy()
        lr = 1e-2
        for _ in range(50000):
            grad = -2 * XD.T.dot(YT - XD.dot(w)) + 2*ridge*w
            w = _simplex_project(w - lr*grad)
        weights = w
    else:
        weights = res.x

    w_series = pd.Series(weights, index=donors, name="weight").sort_values(ascending=False)
    y_synth = y_df[donors].dot(w_series)
    pre_fit = YT - XD.dot(weights)
    pre_rmspe = float(np.sqrt(np.mean(pre_fit**2)))
    return {"weights": w_series, "y_synth": y_synth, "pre_rmspe": pre_rmspe}

def rmspe(y_true: pd.Series, y_hat: pd.Series, years):
    e = (y_true.loc[years] - y_hat.loc[years]).to_numpy()
    return float(np.sqrt(np.mean(e**2)))

def placebo_in_space(y_df, treated, donors, event_year, pre_years, post_years, min_pre_fit_rank=0.5):
    records = []
    for unit in [treated] + donors:
        dset_donors = [d for d in y_df.columns if d != unit]
        pre = fit_scm(y_df, unit, dset_donors, pre_years)
        y_synth = pre["y_synth"]
        pre_r = pre["pre_rmspe"]
        post_r = rmspe(y_df[unit], y_synth, post_years)
        ratio = post_r / (pre_r + 1e-12)
        gaps = y_df[unit] - y_synth
        records.append({
            "unit": unit, "pre_rmspe": pre_r, "post_rmspe": post_r,
            "ratio": ratio, "max_abs_gap": float(np.abs(gaps.loc[post_years]).max()),
            "gaps": gaps
        })
    df = pd.DataFrame(records)
    thr = df.loc[df["unit"] != treated, "pre_rmspe"].quantile(min_pre_fit_rank)
    return df[df["pre_rmspe"] <= thr].reset_index(drop=True)

def leave_one_out(y_df, treated, donors, pre_years):
    base = fit_scm(y_df, treated, donors, pre_years)
    results = {"base": base}
    for d in base["weights"].index:
        alt_donors = [x for x in donors if x != d]
        try:
            results[f"drop_{d}"] = fit_scm(y_df, treated, alt_donors, pre_years)
        except Exception as e:
            results[f"drop_{d}"] = {"error": str(e)}
    return results
