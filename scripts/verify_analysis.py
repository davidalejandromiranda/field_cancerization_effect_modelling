"""Numerical smoke tests for the supporting-information analysis."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scr import data, methods


dense_time = np.linspace(1.0, 13.0, 100)

# Algebraically equivalent baseline parameterizations must agree numerically.
baseline_original = methods.predict_breast(
    methods.baseline_original_rhs, data.BASELINE_ORIGINAL_PARAMS, times=dense_time
)
baseline_reparam = methods.predict_breast(
    methods.baseline_reparam_rhs, data.BASELINE_REPARAM_PARAMS, times=dense_time
)
np.testing.assert_allclose(baseline_original, baseline_reparam, rtol=1e-10, atol=1e-10)

# Every stored cross-cancer trajectory and dataset must be finite and loadable.
for case in data.TUMOR_CASES:
    observed_time, observed = methods.load_tumor_case(case)
    model_time, predicted = methods.predict_tumor_case(case)
    assert observed_time.size == observed.size > 0
    assert model_time.size == predicted.size > 0
    assert np.isfinite(observed).all() and np.isfinite(predicted).all()

# Cross-cancer panels are consistently expressed in mm^3. Breast is the only
# dataset requiring conversion from stored cm^3 and from a cell-state model.
breast_case = next(case for case in data.TUMOR_CASES
                   if case["name"] == "Breast cancer")
_, breast_observed_cm3 = methods.load_tumor_case(breast_case)
assert breast_observed_cm3[0] * breast_case["observed_to_mm3"] == 400.0
assert breast_case["model_to_mm3"] == 1000.0 / data.CELL_DENSITY
assert all(case["observed_to_mm3"] == 1.0
           for case in data.TUMOR_CASES if case is not breast_case)

# The reported model ranking must remain stable under the current data choice.
comparison = methods.model_comparison_table()
assert comparison["AIC"].idxmin() == "Extended FCE (reparameterized)"
assert comparison["BIC"].idxmin() == "Extended FCE (reparameterized)"

# Admissible sampling must be deterministic and remain inside its intervals.
_, samples_a, predictions_a = methods.sample_admissible_predictions(20)
_, samples_b, predictions_b = methods.sample_admissible_predictions(20)
np.testing.assert_array_equal(samples_a, samples_b)
np.testing.assert_array_equal(predictions_a, predictions_b)
assert (samples_a >= data.EXTENDED_INITIAL_INTERVALS[:, 0]).all()
assert (samples_a <= data.EXTENDED_INITIAL_INTERVALS[:, 1]).all()

print("All numerical verification checks passed")
