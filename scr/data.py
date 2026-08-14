"""Data and reported parameter values used by the analysis notebook."""

from pathlib import Path

import numpy as np
import pandas as pd

CELL_DENSITY = 1.0e7  # cells / cm^3
DATA_DIR = Path(__file__).resolve().parent / "tipos de cancer"

# Breast-cancer observations reported in the manuscript (days, cm^3).
BREAST_TIME = np.arange(1.0, 14.0)
BREAST_VOLUME = np.array(
    [0.40, 0.65, 0.79, 0.93, 1.06, 1.21, 1.39,
     1.57, 1.85, 2.21, 2.50, 2.76, 3.03], dtype=float
)
BREAST_INITIAL_CONDITIONS = np.array([0.40e7, 0.40e7])

# Table 3. Order: r, 1/K, aK_T, lambda_ST, lambda_N, gamma, gamma_prime.
EXTENDED_INITIAL_PARAMS = np.array(
    [0.3913, 1.0 / 150000000.02548113, 1.675, 1.1835,
     2.1162, 1.824, 0.8135]
)
EXTENDED_INITIAL_INTERVALS = np.array([
    (0.3910, 0.3916),
    (1.0 / 150000000.0308436, 1.0 / 150000000.02011868),
    (1.670, 1.680),
    (1.181, 1.186),
    (2.1160, 2.1164),
    (1.821, 1.827),
    (0.813, 0.814),
])

# Values used to reproduce Table 5.
BASELINE_ORIGINAL_PARAMS = np.array([1.396, 1.0 / 1.5e8, 1.645, 1.113])
BASELINE_REPARAM_PARAMS = np.array([
    BASELINE_ORIGINAL_PARAMS[0] - BASELINE_ORIGINAL_PARAMS[2]
    + BASELINE_ORIGINAL_PARAMS[3] / 2.0,
    BASELINE_ORIGINAL_PARAMS[0] * BASELINE_ORIGINAL_PARAMS[1],
])
EXTENDED_REPARAM_PARAMS = np.array([-0.6915, 2.61e-9, 2.1162, 1.824, 0.8135])

PARAMETER_SYMBOLS = ["r", "1/K", "aK_T", "lambda_ST", "lambda_N", "gamma", "gamma_prime"]
PARAMETER_UNITS = ["day^-1", "cell^-1"] + 5 * ["day^-1"]


def load_breast_data():
    """Return the breast observations as a new DataFrame and arrays."""
    frame = pd.DataFrame({"time": BREAST_TIME, "volume": BREAST_VOLUME})
    return frame, BREAST_TIME.copy(), BREAST_VOLUME.copy()


# Backward-compatible names from the public v1 repository.
df, time, values = load_breast_data()
initial_conditions = BREAST_INITIAL_CONDITIONS
p0 = EXTENDED_INITIAL_PARAMS
intervals = [tuple(row) for row in EXTENDED_INITIAL_INTERVALS]
r, x2, aK_T, lambda_ST, lambda_N, gamma, gamma_prime = p0
K = 1.0 / x2
parameter_symbols = PARAMETER_SYMBOLS
parameter_names = PARAMETER_SYMBOLS
parameter_units = PARAMETER_UNITS
ranges_df = pd.DataFrame({
    "Parameter": PARAMETER_SYMBOLS,
    "Range": [row.copy() for row in EXTENDED_INITIAL_INTERVALS],
})


# The five non-breast fits retain their legacy reported volume scales. The
# manuscript's claim that every dataset was converted to cells needs author
# confirmation before these values can be transformed safely.
TUMOR_CASES = (
    {"name": "Gastric cancer", "file": DATA_DIR / "cancer gastrico" / "datos.xlsx",
     "params": (0.7027, 1.5e8, 1.350, 1.342, 0.6490, 0.4438, 0.005463),
     "initial": (1.2, 1.2), "t_max": 16.0,
     "observed_to_mm3": 1.0, "model_to_mm3": 1.0},
    {"name": "Pancreatic cancer", "file": DATA_DIR / "cancer de pancreas" / "datos.xlsx",
     "params": (0.3899, 1.5e8, 1.683, 1.183, 1.503, 1.657, 0.8515),
     "initial": (96.0, 96.0), "t_max": 26.0,
     "observed_to_mm3": 1.0, "model_to_mm3": 1.0},
    {"name": "Breast cancer", "file": DATA_DIR / "cancer de mama" / "uso.xlsx",
     "params": (0.2828, 1.5e8, 1.388, 1.005, 1.728, 0.4343, 0.2563),
     "initial": (0.0, 0.4e7), "t_max": 15.0,
     "observed_to_mm3": 1000.0,
     "model_to_mm3": 1000.0 / CELL_DENSITY},
    {"name": "Colon cancer", "file": DATA_DIR / "cancer colon" / "datos2.xlsx",
     "params": (0.895581739927349, 6752.19142827617, 3.56658300529841,
                5.69922088556434, 0.135623851069622, 9.99389579975293,
                4.81000915058404),
     "initial": (7.67567060905876, 7.67567060905876), "t_max": 32.0,
     "observed_to_mm3": 1.0, "model_to_mm3": 1.0},
    {"name": "Esophageal cancer", "file": DATA_DIR / "cancer de esofago" / "datos.xlsx",
     "params": (0.2045, 1.5e8, 1.658, 1.061, 0.4972, 0.8910, 2.170),
     "initial": (91.0, 91.0), "t_max": 23.0,
     "observed_to_mm3": 1.0, "model_to_mm3": 1.0},
    {"name": "Skin cancer", "file": DATA_DIR / "Cancer de piel" / "cancer piel 1.xlsx",
     "params": (0.07201, 1.5e8, 1.994, 1.024, 1.807, 1.488, 1.560),
     "initial": (31.3, 31.3), "t_max": 19.0,
     "observed_to_mm3": 1.0, "model_to_mm3": 1.0},
)

# Scenario fits displayed in manuscript Figure 5. These reproduce the figure's
# 0.4e6 values (which conflict with the 0.1e7 values printed in Table 2).
INITIAL_CONDITION_SCENARIOS = (
    {"label": r"$C_0=0,\ N_0=0.4\times10^7$", "initial": (0.0, 0.4e7),
     "fixed_params": (-0.07752, 1.0 / 1.5e8, 2.468, 2.534, 2.642, 1.003, 0.6809)},
    {"label": r"$C_0=0.4\times10^6,\ N_0=0.4\times10^7$", "initial": (0.4e6, 0.4e7),
     "fixed_params": (0.3913, 1.0 / 555105303.476, 1.675, 1.1835, 2.1162, 1.824, 0.8135)},
    {"label": r"$C_0=0.4\times10^7,\ N_0=0.4\times10^6$", "initial": (0.4e7, 0.4e6),
     "fixed_params": (0.3913, 1.0 / 308175906.808, 1.675, 1.1835, 2.1162, 1.824, 0.8135)},
    {"label": r"$C_0=0.4\times10^7,\ N_0=0$", "initial": (0.4e7, 0.0),
     "fixed_params": (1.690, 1.0 / 1.5e8, 2.606, 2.047, 0.2092, 2.267, 2.589)},
)
SHIFTED_INITIAL_TIME = -0.2162
SHIFTED_SCENARIO_PARAMS = np.array(
    [0.2690, 1.0 / 1.5e8, 0.8883, 1.568, 1.450, 1.077, 0.003246]
)
