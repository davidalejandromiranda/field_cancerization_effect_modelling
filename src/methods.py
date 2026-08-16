"""Numerical models, diagnostics, and figures used by the notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping

from src import data

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


def original_to_reparam_extended(params):
    """Return (m1, m2, lambda_N, gamma, gamma_prime) from the original form."""
    r, inv_k, a_k_t, lambda_st, lambda_n, gamma, gamma_prime = np.asarray(params, dtype=float)
    return np.array([
        r - a_k_t + lambda_st / 2.0,
        r * inv_k,
        lambda_n,
        gamma,
        gamma_prime,
    ])


def rescale_reparam_extended(params, state_scale):
    """Apply C_V=C/s, N_V=N/s to the reparameterized extended system."""
    m1, m2, lambda_n, gamma, gamma_prime = np.asarray(params, dtype=float)
    return np.array([m1, m2 * float(state_scale), lambda_n, gamma, gamma_prime])


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


def predict_breast_from_initial_time(rhs, params, initial, initial_time, observation_times):
    """Predict breast tumor volume when the initial state is set before observations."""
    evaluation_times = np.concatenate((
        [float(initial_time)], np.asarray(observation_times, dtype=float)
    ))
    prediction = _solve(rhs, params, initial, evaluation_times).y[0] / data.CELL_DENSITY
    return prediction[1:]


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

FIT_PARAMETER_NAMES = {
    2: ("m1", "m2"),
    5: ("m1", "m2", "lambda_N", "gamma", "gamma_prime"),
    7: ("r", "1/K", "aK_T", "lambda_ST", "lambda_N", "gamma", "gamma_prime"),
}


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


def fit_result_table(result, parameter_names=None):
    """Return a DataFrame of fitted estimates using manuscript notation."""
    estimates = np.asarray(result.x, dtype=float)
    if parameter_names is None:
        parameter_names = FIT_PARAMETER_NAMES.get(estimates.size)
    if parameter_names is None or len(parameter_names) != estimates.size:
        raise ValueError(
            "parameter_names must match the number of fitted parameters."
        )
    return pd.DataFrame({
        "parameter": tuple(parameter_names),
        "estimate": estimates,
    })


def format_fit_result(result, model_name="Model fit", parameter_names=None):
    """Format a scipy basinhopping result using readable manuscript notation."""
    table = fit_result_table(result, parameter_names)
    lines = [
        model_name,
        f"MSE = {float(result.fun):.12g}",
        "Parameter estimates:",
    ]
    for row in table.itertuples(index=False):
        lines.append(f"  {row.parameter} = {row.estimate:.12g}")
    return "\n".join(lines)


def print_fit_result(result, model_name="Model fit", parameter_names=None):
    """Print and return the human-readable fitted-parameter summary."""
    text = format_fit_result(result, model_name, parameter_names)
    print(text)
    return text


def fitting_smoke_test():
    """Run a small deterministic optimization to verify the fitting API."""
    initial_guess = data.BASELINE_REPARAM_PARAMS * np.array([1.05, 0.95])
    bounds = [(-2.0, 2.0), (0.0, 1e-7)]
    initial = data.BREAST_INITIAL_CONDITIONS[:1]
    start_mse = objective(
        initial_guess, baseline_reparam_rhs, initial, data.BREAST_TIME, data.BREAST_VOLUME
    )
    result = fit_model(
        baseline_reparam_rhs,
        initial_guess=initial_guess,
        bounds=bounds,
        initial=initial,
        niter=1,
    )
    stored_mse = objective(
        data.BASELINE_REPARAM_PARAMS,
        baseline_reparam_rhs,
        initial,
        data.BREAST_TIME,
        data.BREAST_VOLUME,
    )
    return {
        "start_MSE": float(start_mse),
        "smoke_test_MSE": float(result.fun),
        "stored_baseline_reparam_MSE": float(stored_mse),
        "improved_from_start": bool(np.isfinite(result.fun) and result.fun < start_mse),
        "parameters": np.asarray(result.x, dtype=float).tolist(),
    }


def admissible_parameter_sample(candidate_count=None, seed=RNG_SEED):
    """Sample candidates and retain only those satisfying MSE <= epsilon."""
    if candidate_count is None:
        candidate_count = data.ADMISSIBLE_CANDIDATE_COUNT
    rng = np.random.default_rng(seed)
    low, high = data.EXTENDED_INITIAL_INTERVALS.T
    candidates = rng.uniform(low, high, size=(candidate_count, len(low)))
    retained = []
    retained_mse = []
    for params in candidates:
        prediction = predict_breast(extended_original_rhs, params, times=data.BREAST_TIME)
        candidate_mse = mse(data.BREAST_VOLUME, prediction)
        if candidate_mse <= data.ADMISSIBLE_MSE_EPSILON:
            retained.append(params)
            retained_mse.append(candidate_mse)
    if not retained:
        raise RuntimeError(
            "No admissible parameter vectors retained; increase candidate_count "
            "or verify ADMISSIBLE_MSE_EPSILON."
        )
    return candidates, np.asarray(retained), np.asarray(retained_mse)


def sample_admissible_predictions(candidate_count=None, seed=RNG_SEED, times=None):
    """Return predictions from vectors satisfying MSE <= ADMISSIBLE_MSE_EPSILON."""
    if times is None:
        times = np.linspace(data.BREAST_TIME[0], data.BREAST_TIME[-1], 500)
    _, samples, retained_mse = admissible_parameter_sample(candidate_count, seed)
    predictions = np.vstack([
        predict_breast(extended_original_rhs, params, times=times)
        for params in samples
    ])
    return np.asarray(times), samples, predictions, retained_mse


def plot_breast_comparison(candidate_count=None, seed=RNG_SEED):
    """Create the four numerical panels requested for manuscript Figure 3."""
    dense_time = np.linspace(1.0, 13.0, 500)
    baseline = predict_breast(baseline_original_rhs, data.BASELINE_ORIGINAL_PARAMS, times=dense_time)
    extended = predict_breast(extended_original_rhs, data.EXTENDED_INITIAL_PARAMS, times=dense_time)
    sample_time, retained_params, sampled, _ = sample_admissible_predictions(
        candidate_count, seed, dense_time
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), dpi=150)

    axes[0, 0].plot(dense_time, baseline, color="black", label="Baseline")
    axes[0, 0].scatter(data.BREAST_TIME, data.BREAST_VOLUME, color="red", s=25, label="Data")
    axes[0, 0].set_title("Baseline model")

    axes[0, 1].fill_between(sample_time, sampled.min(axis=0), sampled.max(axis=0),
                            color="green", alpha=0.2, label="Admissible region")
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
        for params in retained_params
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
    frame = pd.read_excel(Path(case["file"]), usecols=["time", "volume"]).dropna()
    return (frame["time"].to_numpy(dtype=float),
            frame["volume"].to_numpy(dtype=float))


def _volume_scale_case_rhs(t, y, r, carrying_capacity, a_k_t,
                           lambda_st, lambda_n, gamma, gamma_prime):
    """Legacy scale used by the reported exploratory cross-cancer fits."""
    return extended_original_rhs(t, y, r, 1.0 / carrying_capacity,
                                 a_k_t, lambda_st, lambda_n, gamma, gamma_prime)


def predict_tumor_case(case, points=500):
    initial_time = float(case.get("model_initial_time", 0.0))
    times = np.linspace(initial_time, float(case["t_max"]), points)
    prediction = _solve(_volume_scale_case_rhs, case["params"],
                        case["initial"], times).y[0]
    return times, prediction * case["model_to_plot_scale"]


def _axis_volume_label(unit):
    if unit == "mm^3":
        return r"Volume (mm$^3$)"
    if unit == "cm^3":
        return r"Volume (cm$^3$)"
    return f"Volume ({unit})"


def _format_cell_count(value):
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0**exponent
    return rf"{coefficient:g}\times10^{exponent}"


def initial_condition_label(scenario):
    c0, n0 = scenario["initial"]
    return rf"$C_0={_format_cell_count(c0)},\ N_0={_format_cell_count(n0)}$"


def _scenario_row(scenario, temporal_initialization, params, t0):
    params = np.asarray(params, dtype=float)
    if np.isnan(t0):
        prediction = predict_breast(
            extended_original_rhs, params, initial=scenario["initial"], times=data.BREAST_TIME
        )
    else:
        prediction = predict_breast_from_initial_time(
            extended_original_rhs, params, scenario["initial"], t0, data.BREAST_TIME
        )
    reparam = original_to_reparam_extended(params)
    return {
        "scenario_description": scenario["description"],
        "temporal_initialization": temporal_initialization,
        "C0": scenario["initial"][0],
        "N0": scenario["initial"][1],
        "r": params[0],
        "inv_K": params[1],
        "aK_T": params[2],
        "lambda_ST": params[3],
        "lambda_N": params[4],
        "gamma": params[5],
        "gamma_prime": params[6],
        "m1": reparam[0],
        "m2": reparam[1],
        "t0": t0,
        "MSE": mse(data.BREAST_VOLUME, prediction),
        "parameter_source": "stored optimization-derived value used to reproduce the manuscript analysis",
    }


def initial_condition_parameter_table():
    """Return stored optimization-derived parameters for Figure 5 scenarios."""
    rows = []
    for scenario in data.INITIAL_CONDITION_SCENARIOS:
        rows.append(_scenario_row(
            scenario, "fixed initial time", scenario["fixed_params"], np.nan
        ))
        row = _scenario_row(
            scenario, "optimized effective temporal offset",
            data.SHIFTED_SCENARIO_PARAMS, data.SHIFTED_INITIAL_TIME
        )
        row["parameter_source"] = (
            "stored optimization-derived value used to reproduce the manuscript analysis"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def export_initial_condition_parameters(path="initial_condition_parameter_export.csv"):
    """Write the Figure-5 stored parameter table to CSV and return it."""
    table = initial_condition_parameter_table()
    table.to_csv(path, index=False)
    return table


def plot_cross_cancer(cases=data.TUMOR_CASES):
    """Plot the six independently parameterized exploratory trajectories."""
    colors = ("#377eb8", "#ff7f0e", "#2ca02c",
              "#d62728", "#9467bd", "#8c564b")
    markers = ("o", "s", "^", "D", "v", "p")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), dpi=150)
    for ax, case, color, marker in zip(axes.flat, cases, colors, markers):
        obs_time, observed = load_tumor_case(case)
        model_time, predicted = predict_tumor_case(case)
        observed = observed * case["observed_to_plot_scale"]
        ax.plot(model_time, predicted, color=color, linewidth=2.2)
        ax.scatter(
            obs_time, observed, s=34, marker=marker, color=color,
            edgecolor="black", linewidth=0.7, zorder=3,
        )
        ax.set_title(case["name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (day)", fontsize=11, fontweight="bold")
        ax.set_ylabel(_axis_volume_label(case["plot_unit"]), fontsize=11, fontweight="bold")
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
                     linestyle=style, linewidth=2,
                     label=initial_condition_label(scenario))

        shifted_time = np.linspace(data.SHIFTED_INITIAL_TIME, 13.0, 500)
        shifted_prediction = predict_breast(
            extended_original_rhs, data.SHIFTED_SCENARIO_PARAMS,
            initial=scenario["initial"], times=shifted_time,
        )
        axes[1].plot(shifted_time, shifted_prediction, color=color,
                     linestyle=style, linewidth=2,
                     label=initial_condition_label(scenario))

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
    return plot_breast_comparison(candidate_count=max(num_random, data.ADMISSIBLE_CANDIDATE_COUNT))


def compute_residuals_with_field(params):
    return data.time.copy(), data.values - predict_breast(extended_original_rhs, params)


def compute_residuals_baseline(params):
    return data.time.copy(), data.values - predict_breast(baseline_original_rhs, params)
