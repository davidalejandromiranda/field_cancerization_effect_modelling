"""Build the clean, output-free supporting-information notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    markdown("""# Incorporating the *field cancerization effect* to improve modeling of untreated cancer growth

Supporting-information notebook. It reproduces the numerical analyses described in the manuscript from the data and reported parameter values included in this repository.
"""),
    markdown("""## 1. Environment and reproducibility

All stochastic operations use the fixed seed defined in `scr.methods.RNG_SEED`. Model functions and plotting routines are kept in importable modules so that this notebook remains a concise, executable record of the analysis.
"""),
    code("""from importlib.metadata import version

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from scr import data, methods

print({
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "matplotlib": matplotlib.__version__,
    "seed": methods.RNG_SEED,
})
"""),
    markdown("""## 2. Breast-cancer observations

The principal dataset comprises 13 measurements over days 1–13. Volume is expressed in cm³ and converted internally to cancer-cell population using the stated density of $10^7$ cells/cm³ when solving the models.
"""),
    code("""breast_data, _, _ = data.load_breast_data()
breast_data
"""),
    markdown("""## 3. Mathematical formulations

The implementation contains the original and reparameterized versions of both models:

- Baseline: $dC/dt=rC(1-C/K)-aK_TC+\\lambda_{ST}C/2$.
- Reparameterized baseline: $dC/dt=m_1C-m_2C^2$.
- Extended FCE: the baseline equation plus $\\lambda_NN$, with $dN/dt=\\gamma'C-\\gamma N$.
- Reparameterized FCE: $dC/dt=m_1C-m_2C^2+\\lambda_NN$ and the same equation for $N$.

The original parameterizations are structurally redundant. They are retained only to reproduce the comparisons in the manuscript; inference should use the reparameterized forms.
"""),
    code("""reported_parameters = pd.DataFrame({
    "parameter": data.PARAMETER_SYMBOLS,
    "estimate": data.EXTENDED_INITIAL_PARAMS,
    "lower": data.EXTENDED_INITIAL_INTERVALS[:, 0],
    "upper": data.EXTENDED_INITIAL_INTERVALS[:, 1],
    "unit": data.PARAMETER_UNITS,
})
reported_parameters
"""),
    markdown("""## 4. Breast-cancer comparison and admissible parameter sampling

The green region is computed from trajectories sampled uniformly from the reported admissible parameter intervals. It is not labeled as a confidence or prediction interval because no probabilistic coverage statement has been established.
"""),
    code("""figure_3 = methods.plot_breast_comparison(sample_count=300)
plt.show()
"""),
    markdown("""### Relative residuals and information criteria

RSS, MSE, AIC, and BIC follow equations (10)–(13) of the manuscript. All four models use the same 13 observations and response scale.
"""),
    code("""comparison = methods.model_comparison_table()
comparison.round({"RSS": 6, "MSE": 6, "AIC": 2, "BIC": 2})
"""),
    markdown("""### Audit against manuscript Table 5

The manuscript values are compared explicitly with the values recomputed from the dataset declared in the Methods section ($V(1)=0.40$ cm³). A nonzero difference identifies a manuscript–data inconsistency rather than an execution failure.
"""),
    code("""manuscript_table_5 = pd.DataFrame(
    {
        "MSE": [0.017173, 0.017173, 0.002860, 0.003184],
        "AIC": [-44.84, -48.84, -62.14, -64.74],
        "BIC": [-42.58, -47.71, -58.18, -61.92],
    },
    index=comparison.index,
)
comparison[["MSE", "AIC", "BIC"]].round(6) - manuscript_table_5
"""),
    markdown("""## 5. Exploratory cross-cancer analysis

The following plots reproduce the six independently parameterized trajectories. These heterogeneous datasets assess descriptive flexibility only; they do not validate a universal biological mechanism or a common parameter set.

**Units:** every panel is displayed in mm³. The five non-breast spreadsheets are already used in their reported mm³ scale. Breast observations are stored in cm³ and multiplied by 1000; the breast model state is converted from cells using $V_{mm^3}=C/10^7\times1000$. The legacy reported fits for the five non-breast datasets still operate directly on their volume scales, so the manuscript's statement that every dataset was converted to cell counts must be clarified before submission.
"""),
    code("""figure_4 = methods.plot_cross_cancer()
plt.show()
"""),
    markdown("""## 6. Initial-condition scenarios

This reproduces manuscript Figure 5. The left panel shows the four reported fits when the initial time is fixed at the first observation. The right panel uses the shared reported kinetic parameters and the estimated initial time $t_0=-0.2162$ day. These are reported fitted values, not a new optimization performed by this notebook.
"""),
    code("""figure_initial_conditions = methods.plot_initial_condition_scenarios()
plt.show()
"""),
    markdown("""## 7. Numerical fitting API

`methods.fit_model` provides seeded basin hopping with bounded L-BFGS-B local minimization. Fitting is intentionally not repeated automatically here because the manuscript tables are reproductions from reported estimates and the original parameterizations are non-identifiable. Any new inference run should record its bounds, iteration count, seed, fitted parameters, and objective value.
"""),
    code("""# Example configuration (not run automatically):
# result = methods.fit_model(
#     methods.extended_reparam_rhs,
#     initial_guess=data.EXTENDED_REPARAM_PARAMS,
#     bounds=[(-2.0, 1.0), (0.0, 1e-7), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0)],
#     initial=data.BREAST_INITIAL_CONDITIONS,
#     niter=100,
# )
print("Seeded fitting API available as methods.fit_model")
"""),
    markdown("""## 8. Scope of reproduction

Successful execution verifies that the stored data can be loaded, all four breast-model formulations can be integrated, the reported metrics can be recomputed, and the three analysis figures can be generated. It does not by itself establish biological validity, practical identifiability, or statistical confidence coverage.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"cell-{index:02d}"

Path("simulation_fce_volume_results.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
