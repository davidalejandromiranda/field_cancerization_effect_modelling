"""Numerical models, diagnostics, and figures used by the notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping

from scr import data

RNG_SEED = 20260814


def baseline_original_rhs(t, y, r, inv_k, a_k_t, lambda_st):
    """Baseline equation (2), retaining its redundant parameters."""
    del t
    c = y[0]
    return [r * c * (1.0 - inv_k * c) - a_k_t * c + lambda_st * c / 2.0]


def baseline_reparam_rhs(t, y, m1, m2):
    """Reparameterized baseline equation (4)."""
    del t
    c = y[0]
    return [m1 * c - m2 * c**2]


def extended_original_rhs(t, y, r, inv_k, a_k_t, lambda_st, lambda_n, gamma, gamma_prime):
    """Extended equation (6), using inverse carrying capacity 1/K."""
    del t
    c, n = y
    dc_dt = r * c * (1.0 - inv_k * c) - a_k_t * c + lambda_st * c / 2.0 + lambda_n * n
    dn_dt = gamma_prime * c - gamma * n
    return [dc_dt, dn_dt]


def extended_reparam_rhs(t, y, m1, m2, lambda_n, gamma, gamma_prime):
    """Reparameterized extended equation (8)."""
    del t
    c, n = y
    return [m1 * c - m2 * c**2 + lambda_n * n, gamma_prime * c - gamma * n]


# Compatibility aliases used by the public v1 repository.
cancer_model = extended_original_rhs
cancer_model2 = baseline_original_rhs


def _solve(rhs, params, initial, times):
    times = np.asarray(times, dtype=float)
    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        np.asarray(initial, dtype=float),
        args=tuple(np.asarray(params, dtype=float)),
        t_eval=times,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success or solution.y.shape[1] != len(times):
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution


def predict_breast(rhs, params, initial=None, times=None):
    """Predict breast tumor volume in cm^3 at the requested times."""
    if times is None:
        times = data.BREAST_TIME
    if initial is None:
        initial = (data.BREAST_INITIAL_CONDITIONS[:1]
                   if rhs in (baseline_original_rhs, baseline_reparam_rhs)
                   else data.BREAST_INITIAL_CONDITIONS)
    return _solve(rhs, params, initial, times).y[0] / data.CELL_DENSITY


def mse(observed, predicted):
    return float(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2))


def relative_residuals(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    return (observed - np.asarray(predicted, dtype=float)) / observed


def information_criteria(observed, predicted, parameter_count):
    """Return RSS, MSE, AIC, and BIC as in manuscript equations (10)-(13)."""
    residual = np.asarray(observed) - np.asarray(predicted)
    rss = float(residual @ residual)
    n_obs = residual.size
    model_mse = rss / n_obs
    aic = n_obs * np.log(model_mse) + 2 * parameter_count
    bic = n_obs * np.log(model_mse) + parameter_count * np.log(n_obs)
    return {"RSS": rss, "MSE": model_mse, "AIC": aic, "BIC": bic}


MODEL_SPECIFICATIONS = (
    ("Baseline (original)", baseline_original_rhs, data.BASELINE_ORIGINAL_PARAMS, 4),
    ("Baseline (reparameterized)", baseline_reparam_rhs, data.BASELINE_REPARAM_PARAMS, 2),
    ("Extended FCE (initial)", extended_original_rhs, data.EXTENDED_INITIAL_PARAMS, 7),
    ("Extended FCE (reparameterized)", extended_reparam_rhs, data.EXTENDED_REPARAM_PARAMS, 5),
)


def model_comparison_table():
    """Compute the four-model comparison underlying manuscript Table 5."""
    rows = []
    for name, rhs, params, parameter_count in MODEL_SPECIFICATIONS:
        prediction = predict_breast(rhs, params)
        rows.append({"Model": name, "k": parameter_count,
                     **information_criteria(data.BREAST_VOLUME, prediction, parameter_count)})
    return pd.DataFrame(rows).set_index("Model")


def objective(params, rhs, initial, times, observations):
    """MSE objective with integration failures mapped to a large penalty."""
    params = np.asarray(params, dtype=float)
    if not np.all(np.isfinite(params)):
        return 1e30
    try:
        prediction = _solve(rhs, params, initial, times).y[0] / data.CELL_DENSITY
    except (RuntimeError, ValueError, FloatingPointError):
        return 1e30
    return mse(observations, prediction)


def fit_model(rhs, initial_guess, bounds, initial, times=None,
              observations=None, niter=100, seed=RNG_SEED):
    """Fit a model with deterministic basin hopping and bounded local steps."""
    if times is None:
        times = data.BREAST_TIME
    if observations is None:
        observations = data.BREAST_VOLUME
    minimizer = {"method": "L-BFGS-B", "bounds": bounds,
                 "args": (rhs, initial, times, observations)}
    return basinhopping(
        objective, np.asarray(initial_guess, dtype=float), niter=niter,
        minimizer_kwargs=minimizer, seed=seed,
    )


def sample_admissible_predictions(sample_count=500, seed=RNG_SEED, times=None):
    """Sample the reported intervals and return predicted breast volumes."""
    if times is None:
        times = np.linspace(data.BREAST_TIME[0], data.BREAST_TIME[-1], 500)
    rng = np.random.default_rng(seed)
    low, high = data.EXTENDED_INITIAL_INTERVALS.T
    samples = rng.uniform(low, high, size=(sample_count, len(low)))
    predictions = np.vstack([
        predict_breast(extended_original_rhs, params, times=times)
        for params in samples
    ])
    return np.asarray(times), samples, predictions


def plot_breast_comparison(sample_count=300, seed=RNG_SEED):
    """Create the four numerical panels requested for manuscript Figure 3."""
    dense_time = np.linspace(1.0, 13.0, 500)
    baseline = predict_breast(baseline_original_rhs, data.BASELINE_ORIGINAL_PARAMS, times=dense_time)
    extended = predict_breast(extended_original_rhs, data.EXTENDED_INITIAL_PARAMS, times=dense_time)
    sample_time, _, sampled = sample_admissible_predictions(sample_count, seed, dense_time)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), dpi=150)

    axes[0, 0].plot(dense_time, baseline, color="black", label="Baseline")
    axes[0, 0].scatter(data.BREAST_TIME, data.BREAST_VOLUME, color="red", s=25, label="Data")
    axes[0, 0].set_title("Baseline model")

    axes[0, 1].fill_between(sample_time, sampled.min(axis=0), sampled.max(axis=0),
                            color="green", alpha=0.2, label="Sampled projections")
    axes[0, 1].plot(dense_time, extended, color="black", label="Extended FCE")
    axes[0, 1].scatter(data.BREAST_TIME, data.BREAST_VOLUME,
                       color="red", s=25, label="Data")
    axes[0, 1].set_title("Extended model")

    baseline_obs = predict_breast(baseline_original_rhs, data.BASELINE_ORIGINAL_PARAMS)
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].scatter(
        data.BREAST_TIME,
        relative_residuals(data.BREAST_VOLUME, baseline_obs),
        color="blue", edgecolor="black", linewidth=0.5,
    )
    axes[1, 0].set_title("Baseline relative residuals")

    observed_samples = np.vstack([
        predict_breast(extended_original_rhs, params)
        for params in sample_admissible_predictions(sample_count, seed, data.BREAST_TIME)[1]
    ])
    axes[1, 1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    for residual in (
        data.BREAST_VOLUME[None, :] - observed_samples
    ) / data.BREAST_VOLUME[None, :]:
        axes[1, 1].scatter(
            data.BREAST_TIME, residual, color="blue", edgecolor="black",
            linewidth=0.25, s=14, alpha=0.08,
        )
    axes[1, 1].set_title("Extended-model relative residuals")

    panel_labels = ("a)", "b)", "c)", "d)")
    for index, (label, ax) in enumerate(zip(panel_labels, axes.flat)):
        ax.set_xlabel("Time (days)")
        if index < 2:
            ax.set_ylabel(r"Tumor volume (cm$^3$)")
        else:
            ax.set_ylabel("Relative residual")
            ax.set_ylim(-0.15, 0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.16, 1.04, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold")
        if index < 2:
            ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def load_tumor_case(case):
    frame = pd.read_excel(Path(case["file"]), usecols=["tiempo", "valores"]).dropna()
    return (frame["tiempo"].to_numpy(dtype=float),
            frame["valores"].to_numpy(dtype=float))


def _volume_scale_case_rhs(t, y, r, carrying_capacity, a_k_t,
                           lambda_st, lambda_n, gamma, gamma_prime):
    """Legacy scale used by the reported exploratory cross-cancer fits."""
    return extended_original_rhs(t, y, r, 1.0 / carrying_capacity,
                                 a_k_t, lambda_st, lambda_n, gamma, gamma_prime)


def predict_tumor_case(case, points=500):
    times = np.linspace(0.0, float(case["t_max"]), points)
    prediction = _solve(_volume_scale_case_rhs, case["params"],
                        case["initial"], times).y[0]
    return times, prediction * case["model_to_mm3"]


def plot_cross_cancer(cases=data.TUMOR_CASES):
    """Plot the six independently parameterized exploratory trajectories."""
    colors = ("#377eb8", "#ff7f0e", "#2ca02c",
              "#d62728", "#9467bd", "#8c564b")
    markers = ("o", "s", "^", "D", "v", "p")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), dpi=150)
    for ax, case, color, marker in zip(axes.flat, cases, colors, markers):
        obs_time, observed = load_tumor_case(case)
        model_time, predicted = predict_tumor_case(case)
        observed = observed * case["observed_to_mm3"]
        ax.plot(model_time, predicted, color=color, linewidth=2.2)
        ax.scatter(
            obs_time, observed, s=34, marker=marker, color=color,
            edgecolor="black", linewidth=0.7, zorder=3,
        )
        ax.set_title(case["name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (day)", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"Volume (mm$^3$)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, case["t_max"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.4)
        ax.tick_params(direction="out", width=1.2, length=3.5, labelsize=9)
    fig.tight_layout(h_pad=1.2, w_pad=1.3)
    return fig


def plot_initial_condition_scenarios():
    """Reproduce the two optimization-scheme panels shown in Figure 5."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150,
                             sharex=True, sharey=True)
    color = "#a0004f"
    styles = ("-", "--", "-.", ":")

    for scenario, style in zip(data.INITIAL_CONDITION_SCENARIOS, styles):
        fixed_time = np.linspace(1.0, 13.0, 500)
        fixed_prediction = predict_breast(
            extended_original_rhs, scenario["fixed_params"],
            initial=scenario["initial"], times=fixed_time,
        )
        axes[0].plot(fixed_time, fixed_prediction, color=color,
                     linestyle=style, linewidth=2, label=scenario["label"])

        shifted_time = np.linspace(data.SHIFTED_INITIAL_TIME, 13.0, 500)
        shifted_prediction = predict_breast(
            extended_original_rhs, data.SHIFTED_SCENARIO_PARAMS,
            initial=scenario["initial"], times=shifted_time,
        )
        axes[1].plot(shifted_time, shifted_prediction, color=color,
                     linestyle=style, linewidth=2, label=scenario["label"])

    for label, ax in zip(("a)", "b)"), axes):
        ax.scatter(data.BREAST_TIME, data.BREAST_VOLUME,
                   color="red", s=25, zorder=5)
        ax.set(xlabel="Time (day)", ylabel=r"Tumor volume (cm$^3$)",
               xlim=(-1.0, 14.0), ylim=(0.0, 4.0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.16, 1.04, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


# Public-v1 convenience wrappers.
def OF(params, t_exp=data.time, v_exp=data.values):
    return objective(params, extended_original_rhs, data.initial_conditions,
                     t_exp, v_exp)


def OF2(params, t_exp=data.time, v_exp=data.values):
    return objective(params, baseline_original_rhs, data.initial_conditions[:1],
                     t_exp, v_exp)


def plot_model(params):
    dense = np.linspace(1.0, 13.0, 500)
    plt.plot(dense, predict_breast(extended_original_rhs, params, times=dense),
             color="black")
    plt.scatter(data.time, data.values, color="red")
    plt.xlabel("Time (days)")
    plt.ylabel(r"Tumor volume (cm$^3$)")
    return plt.gcf()


def plot_pei_model(params):
    dense = np.linspace(1.0, 13.0, 500)
    plt.plot(dense, predict_breast(baseline_original_rhs, params, times=dense),
             color="black")
    plt.scatter(data.time, data.values, color="red")
    plt.xlabel("Time (days)")
    plt.ylabel(r"Tumor volume (cm$^3$)")
    return plt.gcf()


def plot_models(params, intervals, num_random=5):
    del params, intervals
    return plot_breast_comparison(sample_count=max(num_random, 1))


def compute_residuals_with_field(params):
    return data.time.copy(), data.values - predict_breast(extended_original_rhs, params)


def compute_residuals_baseline(params):
    return data.time.copy(), data.values - predict_breast(baseline_original_rhs, params)
