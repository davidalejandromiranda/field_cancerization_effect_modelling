import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping, minimize, least_squares
import random

from scr.data import (
    df,
    time,
    values,
    initial_conditions,
    intervals,
    x2,
    aK_T,
    r,
    lambda_ST,
    K,
    lambda_N,
    gamma,
    gamma_prime,
    p0,
    parameter_symbols,
    parameter_names,
    parameter_units,
    ranges_df,
)

__all__ = [
    "df",
    "time",
    "values",
    "initial_conditions",
    "intervals",
    "x2",
    "aK_T",
    "r",
    "lambda_ST",
    "K",
    "lambda_N",
    "gamma",
    "gamma_prime",
    "p0",
    "parameter_symbols",
    "parameter_names",
    "parameter_units",
    "ranges_df",
    "pick_random_numbers",
    "print_parameters",
    "cancer_model",
    "cancer_model2",
    "OF",
    "OF2",
    "plot_models",
    "plot_model",
    "plot_pei_model",
    "compute_residuals_with_field",
    "compute_residuals_baseline",
]


def pick_random_numbers(intervals):
    """Sample one random value from each interval in the provided list."""
    return [random.uniform(low, high) for low, high in intervals]


def print_parameters(params, symbols, units, names=None):
    """Print a parameter vector with associated symbol labels, parameter names, and units."""
    for i in range(len(params)):
        if names is not None:
            print(f"{symbols[i]} = {names[i]} = {params[i]:.6g} {units[i]}")
        else:
            print(f"{symbols[i]} = {params[i]:.6g} {units[i]}")
    print("-" * 40)


def cancer_model(t, y, r, x2, aK_T, lambda_ST, lambda_N, gamma, gamma_prime):
    """Define the cancer growth model with the field cancerization effect.

    The parameter x2 corresponds to 1/K, so the logistic term is
    r C (1 - x2 C) in agreement with the paper notation.
    """
    C, N = y
    dN_dt = -gamma * N + gamma_prime * C
    dC_dt = (
        r * C * (1.0 - x2 * C)
        - aK_T * C
        + (lambda_ST * C) / 2.0
        + lambda_N * N
    )
    return [dC_dt, dN_dt]


def cancer_model2(t, y, r, x2, aK_T, lambda_ST):
    """Define the baseline cancer growth model without the field effect."""
    C = y[0] if isinstance(y, (list, np.ndarray)) else y
    dC_dt = (
        r * C * (1.0 - x2 * C)
        - aK_T * C
        + (lambda_ST * C) / 2.0
    )
    return [dC_dt]



def OF(p, t_exp=time, v_exp=values):
    """Objective function for the model with the field cancerization effect."""
    p = np.asarray(p)
    if np.any(p < 0):
        return 1e9
    C_theo = solve_ivp(
        cancer_model,
        (t_exp[0], t_exp[-1]),
        initial_conditions,
        args=tuple(p),
        dense_output=True,
        t_eval=t_exp
    )
    v_theo = C_theo.y[0] / 1e7
    return np.sum((1 - v_exp / v_theo) ** 2)


def OF2(p, t_exp=time, v_exp=values):
    """Objective function for the model without the field cancerization effect."""
    p = np.asarray(p)
    if np.any(p < 0):
        return 1e9
    sol = solve_ivp(
        cancer_model2,
        (t_exp[0], t_exp[-1]),
        y0=[initial_conditions[0]],
        args=tuple(p),
        dense_output=True,
        t_eval=t_exp
    )
    v_theo = sol.y[0] / 1e7
    return np.sum((1 - v_exp / v_theo) ** 2)


def plot_models(p0, intervals, num_random=5):
    """Plot the mean model and random simulations within the admissible parameter ranges."""
    t_span = (1, 15)
    t_eval = np.linspace(1, 15, 1000)

    sol_mean = solve_ivp(
        cancer_model,
        t_span,
        initial_conditions,
        args=tuple(p0),
        dense_output=True,
        t_eval=t_eval
    )
    volume_mean = sol_mean.y[0] / 1e7
    range_upper = volume_mean * 1.05
    range_lower = volume_mean * 0.95

    plt.figure(dpi=120)
    plt.fill_between(t_eval, range_lower, range_upper, color='g', alpha=0.2, label='Prediction band')
    plt.plot(t_eval, volume_mean, 'k-', label='Mean model')

    for i in range(num_random):
        p_random = pick_random_numbers(intervals)
        print(f"Simulation {i+1}:")
        print_parameters(p_random, parameter_symbols, parameter_units, parameter_names)

        sol_random = solve_ivp(
            cancer_model,
            t_span,
            initial_conditions,
            args=tuple(p_random),
            dense_output=True,
            t_eval=t_eval
        )
        volume_random = sol_random.y[0] / 1e7
        if i == 0:
            plt.plot(t_eval, volume_random, ':', lw=1, color='blue', alpha=0.8, label='Simulation within parameter intervals')
        else:
            plt.plot(t_eval, volume_random, ':', lw=1, color='blue', alpha=0.8)

    plt.plot(time, values, marker='o', linestyle='none', color='r', label='Experimental data')
    plt.xlabel('Time (days)')
    plt.ylabel(r'Cancer cell volume ($cm^3$)')
    plt.title('Simulations with uncertainty')
    plt.legend()
    plt.grid(True, lw=0.5, linestyle=':')
    plt.show()


def plot_model(p):
    """Plot the cancer model with the field cancerization effect against experimental data."""
    r, x2, aK_T, lambda_ST, lambda_N, gamma, gamma_prime = p
    t_span = (1, 15)
    t_eval = np.linspace(1, 15, 1000)

    sol = solve_ivp(
        cancer_model,
        t_span,
        initial_conditions,
        args=(r, x2, aK_T, lambda_ST, lambda_N, gamma, gamma_prime),
        dense_output=True,
        t_eval=t_eval
    )

    t = sol.t
    C = sol.y[0]
    volume = C / 1e7

    plt.figure(dpi=120)
    plt.plot(t, volume, 'k', label='Cancer cell volume')
    plt.plot(time, values, marker='o', linestyle='none', color='r', label='Experimental data')
    plt.xlabel('Time (days)')
    plt.ylabel(r'Cancer cell volume ($cm^3$)')
    plt.title('Cancer volume evolution over time')
    plt.legend()
    plt.grid(True, lw=0.5, linestyle=':')
    plt.show()


def plot_pei_model(p):
    """Plot the baseline cancer model without the field cancerization effect."""
    r, x2, aK_T, lambda_ST = p
    t_span = (1, 15)
    t_eval = np.linspace(1, 15, 1000)

    sol = solve_ivp(
        cancer_model2,
        t_span,
        [initial_conditions[0]],
        args=(r, x2, aK_T, lambda_ST),
        dense_output=True,
        t_eval=t_eval
    )

    t = sol.t
    C = sol.y[0]
    volume = C / 1e7

    plt.figure(dpi=120)
    plt.plot(t, volume, 'k', label='Cancer cell volume')
    plt.plot(time, values, marker='o', linestyle='none', color='r', label='Experimental data')
    plt.xlabel('Time (days)')
    plt.ylabel(r'Cancer cell volume ($cm^3$)')
    plt.title('Cancer volume evolution over time')
    plt.legend()
    plt.grid(True, lw=0.5, linestyle=':')
    plt.show()


# Unused helper functions have been moved to scr/others_non_used.py for review.


def compute_residuals_with_field(p):
    """Compute residuals for the model with the field cancerization effect."""
    sol = solve_ivp(
        cancer_model,
        (1, 15),
        initial_conditions,
        args=p,
        t_eval=np.arange(1, 15)
    )

    C = sol.y[0]
    volume = C / 1e7
    common_times = np.intersect1d(sol.t, time)
    model_common = [volume[list(sol.t).index(tt)] for tt in common_times]
    real_common = [values[list(time).index(tt)] for tt in common_times]
    residuals = np.array(real_common) - np.array(model_common)
    return common_times, residuals


def compute_residuals_baseline(p_simple):
    """Compute residuals for the baseline model without the field cancerization effect."""
    sol = solve_ivp(
        cancer_model2,
        (1, 15),
        [initial_conditions[0]],
        args=p_simple,
        t_eval=np.arange(1, 15)
    )

    C = sol.y[0]
    volume = C / 1e7
    common_times = np.intersect1d(sol.t, time)
    model_common = [volume[list(sol.t).index(tt)] for tt in common_times]
    real_common = [values[list(time).index(tt)] for tt in common_times]
    residuals = np.array(real_common) - np.array(model_common)
    return common_times, residuals
