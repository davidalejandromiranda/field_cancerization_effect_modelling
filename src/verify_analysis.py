"""Numerical smoke tests for the supporting-information analysis."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, methods


dense_time = np.linspace(1.0, 13.0, 100)

# Algebraically equivalent baseline parameterizations must agree numerically.
baseline_original = methods.predict_breast(
    methods.baseline_original_rhs, data.BASELINE_ORIGINAL_PARAMS, times=dense_time
)
baseline_reparam = methods.predict_breast(
    methods.baseline_reparam_rhs, data.BASELINE_REPARAM_PARAMS, times=dense_time
)
np.testing.assert_allclose(baseline_original, baseline_reparam, rtol=1e-10, atol=1e-10)

# The reparameterized extended model follows m1 = r - aK_T + lambda_ST / 2
# and m2 = r/K. Under C_V=C/s, N_V=N/s, m2,V = m2*s.
extended_reparam_from_original = methods.original_to_reparam_extended(
    data.EXTENDED_INITIAL_PARAMS
)
extended_original = methods.predict_breast(
    methods.extended_original_rhs, data.EXTENDED_INITIAL_PARAMS, times=dense_time
)
extended_reparam = methods.predict_breast(
    methods.extended_reparam_rhs, extended_reparam_from_original, times=dense_time
)
np.testing.assert_allclose(extended_original, extended_reparam, rtol=1e-10, atol=1e-10)

volume_scale_params = methods.rescale_reparam_extended(
    extended_reparam_from_original, data.CELL_DENSITY
)
volume_scale_solution = methods._solve(
    methods.extended_reparam_rhs,
    volume_scale_params,
    data.BREAST_INITIAL_CONDITIONS / data.CELL_DENSITY,
    dense_time,
).y[0]
np.testing.assert_allclose(extended_original, volume_scale_solution, rtol=1e-8, atol=1e-8)

# Every stored cross-cancer trajectory and dataset must be finite and loadable.
expected_initial_plot_volume = {
    "Gastric cancer": 120.0,
    "Pancreatic cancer": 96.0,
    "Breast cancer": 400.0,
    "Colon cancer": 90.0,
    "Esophageal cancer": 91.0,
    "Skin cancer": 31.3,
}
expected_first_observation_time = {
    "Gastric cancer": 0.0,
    "Pancreatic cancer": 0.0,
    "Breast cancer": 1.0,
    "Colon cancer": 10.0,
    "Esophageal cancer": 0.0,
    "Skin cancer": 0.25,
}
expected_first_observation_volume = {
    "Gastric cancer": 120.0,
    "Pancreatic cancer": 96.0,
    "Breast cancer": 400.0,
    "Colon cancer": 90.0,
    "Esophageal cancer": 91.0,
    "Skin cancer": 31.3,
}
for case in data.TUMOR_CASES:
    observed_time, observed = methods.load_tumor_case(case)
    model_time, predicted = methods.predict_tumor_case(case)
    assert observed_time.size == observed.size > 0
    assert model_time.size == predicted.size > 0
    assert np.isfinite(observed).all() and np.isfinite(predicted).all()
    for key in (
        "observed_unit", "model_state_unit", "initial_unit", "plot_unit",
        "observed_to_plot_scale", "model_to_plot_scale", "source_publication",
        "figure_panel", "experimental_group", "raw_response", "raw_unit",
        "transformation", "initial_experimental_volume_mm3", "doi",
        "experimental_time_unit", "model_initial_time", "first_observation_time",
        "first_observation_volume_mm3",
    ):
        assert key in case, f"{case['name']} lacks dataset metadata: {key}"
    for value in case.values():
        if isinstance(value, str):
            assert "AUTHOR CONFIRMATION REQUIRED" not in value
    assert case["plot_unit"] == "mm^3"
    assert case["experimental_time_unit"] == "day"
    assert case["observed_to_plot_scale"] > 0.0
    assert case["model_to_plot_scale"] > 0.0
    assert not np.isclose(observed.max() * case["observed_to_plot_scale"], 1.0)
    np.testing.assert_allclose(
        case["initial_experimental_volume_mm3"],
        expected_initial_plot_volume[case["name"]],
    )
    np.testing.assert_allclose(
        observed_time[0],
        expected_first_observation_time[case["name"]],
    )
    np.testing.assert_allclose(
        case["first_observation_time"],
        expected_first_observation_time[case["name"]],
    )
    np.testing.assert_allclose(
        observed[0] * case["observed_to_plot_scale"],
        expected_first_observation_volume[case["name"]],
    )
    np.testing.assert_allclose(
        case["first_observation_volume_mm3"],
        expected_first_observation_volume[case["name"]],
    )

colon_case = next(case for case in data.TUMOR_CASES
                  if case["name"] == "Colon cancer")
assert colon_case["model_initial_time"] == 0.0
np.testing.assert_allclose(colon_case["model_initial_volume_mm3"], 7.67567060905876)
np.testing.assert_allclose(colon_case["initial"], (7.67567060905876, 7.67567060905876))
assert colon_case["initial_experimental_volume_mm3"] == 90.0
assert colon_case["first_observation_time"] == 10.0
assert colon_case["first_observation_volume_mm3"] == 90.0

# Current cross-cancer plotting labels use mm^3. Breast is the only dataset
# requiring a non-1.0 display conversion in the current repository workflow.
breast_case = next(case for case in data.TUMOR_CASES
                   if case["name"] == "Breast cancer")
_, breast_observed_cm3 = methods.load_tumor_case(breast_case)
assert breast_case["observed_unit"] == "cm^3"
assert breast_case["model_state_unit"] == "cells"
assert breast_observed_cm3[0] * breast_case["observed_to_plot_scale"] == 400.0
assert breast_case["model_to_plot_scale"] == 1000.0 / data.CELL_DENSITY
assert breast_case["transformation"] == (
    "volume_mm3 = (cell_count / rho) * 1000, rho = 1e7 cells/cm^3"
)
assert all(case["observed_to_plot_scale"] == 1.0
           for case in data.TUMOR_CASES if case is not breast_case)
assert all(case["model_to_plot_scale"] == 1.0
           for case in data.TUMOR_CASES if case is not breast_case)
assert all(case["model_state_unit"] == "legacy volume scale"
           for case in data.TUMOR_CASES if case is not breast_case)

gastric_case = next(case for case in data.TUMOR_CASES
                    if case["name"] == "Gastric cancer")
gastric_raw = np.genfromtxt(gastric_case["raw_file"], delimiter=",", names=True)
_, gastric_observed = methods.load_tumor_case(gastric_case)
np.testing.assert_allclose(
    gastric_raw["relative_volume"] * gastric_raw["V0_mm3"],
    gastric_observed,
)
assert gastric_case["transformation"] == "absolute_volume_mm3 = relative_volume * 100"
assert set(gastric_raw["V0_mm3"]) == {100.0}

# The reported model ranking must remain stable under the current data choice.
comparison = methods.model_comparison_table()
assert comparison["AIC"].idxmin() == "Extended FCE (reparameterized)"
assert comparison["BIC"].idxmin() == "Extended FCE (reparameterized)"
assert set(["RSS", "MSE", "AIC", "BIC"]).issubset(comparison.columns)

# Admissible sampling must be deterministic and every retained vector must
# satisfy MSE <= epsilon.
_, samples_a, predictions_a, retained_mse_a = methods.sample_admissible_predictions(100)
_, samples_b, predictions_b, retained_mse_b = methods.sample_admissible_predictions(100)
np.testing.assert_array_equal(samples_a, samples_b)
np.testing.assert_array_equal(predictions_a, predictions_b)
np.testing.assert_array_equal(retained_mse_a, retained_mse_b)
assert (samples_a >= data.EXTENDED_INITIAL_INTERVALS[:, 0]).all()
assert (samples_a <= data.EXTENDED_INITIAL_INTERVALS[:, 1]).all()
assert (retained_mse_a <= data.ADMISSIBLE_MSE_EPSILON).all()
figure_3 = methods.plot_breast_comparison(candidate_count=100)
assert len(figure_3.axes) == 4

# Figure 5 labels are generated from the stored numerical initial conditions.
expected_labels = [
    r"$C_0=4\times10^5,\ N_0=4\times10^6$",
    r"$C_0=4\times10^6,\ N_0=4\times10^5$",
    r"$C_0=0,\ N_0=4\times10^6$",
    r"$C_0=4\times10^6,\ N_0=0$",
]
actual_labels = [
    methods.initial_condition_label(scenario)
    for scenario in data.INITIAL_CONDITION_SCENARIOS
]
assert actual_labels == expected_labels

# The optimized-time panel must start from the stored fitted t0, not from the
# visual axis limit of -1 day.
assert data.SHIFTED_INITIAL_TIME == -0.2162
figure = methods.plot_initial_condition_scenarios()
assert len([line for line in figure.axes[0].lines if len(line.get_xdata()) == 500]) == 4
right_panel_lines = [line for line in figure.axes[1].lines if len(line.get_xdata()) == 500]
assert len(right_panel_lines) == 4
assert all(line.get_xdata()[0] == data.SHIFTED_INITIAL_TIME
           for line in right_panel_lines)

parameter_export = methods.export_initial_condition_parameters()
assert len(parameter_export) == 8
assert set(parameter_export["temporal_initialization"]) == {
    "fixed initial time",
    "optimized effective temporal offset",
}
assert parameter_export[["C0", "N0"]].notna().all().all()
fixed_rows = parameter_export[
    parameter_export["temporal_initialization"] == "fixed initial time"
]
optimized_rows = parameter_export[
    parameter_export["temporal_initialization"] == "optimized effective temporal offset"
]
assert fixed_rows[["r", "inv_K", "aK_T", "lambda_ST",
                   "lambda_N", "gamma", "gamma_prime", "MSE"]].notna().all().all()
assert optimized_rows[["r", "inv_K", "aK_T", "lambda_ST",
                       "lambda_N", "gamma", "gamma_prime", "t0", "MSE"]].notna().all().all()
np.testing.assert_allclose(optimized_rows["t0"], data.SHIFTED_INITIAL_TIME)
assert all("stored optimization-derived value" in value
           for value in optimized_rows["parameter_source"])

fitting_smoke_test = methods.fitting_smoke_test()
assert fitting_smoke_test["improved_from_start"]
assert np.isfinite(fitting_smoke_test["smoke_test_MSE"])
assert len(fitting_smoke_test["parameters"]) == 2

class _DummyFitResult:
    x = data.EXTENDED_REPARAM_PARAMS
    fun = comparison.loc["Extended FCE (reparameterized)", "MSE"]


formatted_fit = methods.format_fit_result(
    _DummyFitResult(), model_name="Extended FCE (reparameterized)"
)
for token in ("m1", "m2", "lambda_N", "gamma", "gamma_prime", "MSE"):
    assert token in formatted_fit

print("All numerical verification checks passed")
