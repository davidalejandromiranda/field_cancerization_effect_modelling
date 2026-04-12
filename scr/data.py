import numpy as np
import pandas as pd


def load_pei_data():
    """Load the experimental tumor volume data reported by Pei et al."""
    data = {
        "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "values": [0.44, 0.65, 0.79, 0.93, 1.06, 1.21, 1.39,
                   1.57, 1.85, 2.21, 2.50, 2.76, 3.03]
    }
    df = pd.DataFrame(data)
    time = np.array(df["time"])
    values = np.array(df["values"])
    return df, time, values


df, time, values = load_pei_data()

initial_conditions = [0.4e7, 0.4e7]

# Model parameter values and admissible intervals
r = 0.3913
K = 150000000.02548113
x2 = 1.0 / K

aK_T = 1.675
lambda_ST = 1.1835
lambda_N = 2.1162
gamma = 1.824
gamma_prime = 0.8135

p0 = [r, x2, aK_T, lambda_ST, lambda_N, gamma, gamma_prime]

intervals = [
    (0.3910, 0.3916),
    (1.0 / 150000000.0308436, 1.0 / 150000000.02011868),
    (1.67, 1.68),
    (1.181, 1.186),
    (2.1160, 2.1164),
    (1.821, 1.827),
    (0.813, 0.814)
]

parameter_symbols = ["x₁", "x₂", "x₃", "x₄", "x̂₁", "p₁", "p₂"]
parameter_names = ["r", "1/K", "aK_T", "λ_ST", "λ_N", "γ'", "γ"]
parameter_units = ["(days)⁻¹", "(cells)⁻¹"] + 5 * ["(days)⁻¹"]

ranges_df = pd.DataFrame({
    "Parameter": ["r", "1/K", "aK_T", "λ_ST", "λ_N", "γ'", "γ"],
    "Symbol (𝑃̂ = (R,P))": ["x₁", "x₂", "x₃", "x₄", "x̂₁", "p₁", "p₂"],
    "Description": [
        "Cancer cell proliferation rate",
        "Inverse carrying capacity",
        "Cancer cell elimination by effector T cells",
        "Cancer cell activation by β-catenin",
        "Contribution rate of preconditioned cells to cancer-cell proliferation",
        "Formation rate of preconditioned cells induced by cancer",
        "Death rate of preconditioned cells"
    ],
    "Range": [
        np.array([0.3910, 0.3916]),
        np.array([1.0 / 150000000.0308436, 1.0 / 150000000.02011868]),
        np.array([1.67, 1.68]),
        np.array([1.181, 1.186]),
        np.array([2.1160, 2.1164]),
        np.array([0.813, 0.814]),
        np.array([1.821, 1.827])
    ]
})
