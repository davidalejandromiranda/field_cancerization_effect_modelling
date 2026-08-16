# Supporting Information: Field Cancerization Modeling

This repository contains the code and data used to support the manuscript on field cancerization modeling. It is intended as supplementary material for reproducing the numerical simulations, figures, model-comparison metrics, and diagnostic analyses reported in the manuscript, and for reviewing the mathematical model structure.

## Repository contents

- `simulation_fce_volume_results.ipynb`
  - Jupyter notebook with the main simulations and figures.
- `src/data.py`
  - Experimental tumor volume data and model parameter definitions.
- `src/methods.py`
  - Model equations, numerical integration, fitting, diagnostics, and figures.
- `src/cancer_types/`
  - Tabular observations for the six tumor-growth datasets. Gastric cancer also includes the preserved raw relative-volume extraction used for the explicit `V = V_rel * V0` conversion.
- `src/verify_analysis.py`
  - Numerical verification checks for the analysis workflow.
- `initial_condition_parameter_export.csv`
  - Machine-readable Figure-5 parameter export generated from the stored optimization-derived values.
- `requirements.txt`
  - Reproducible Python dependency ranges.

## Model overview

The model describes tumor volume evolution with an additional field effect state variable. The main variables are:

- `C`: tumor cells.
- `N`: normal preconditioned cells.

The full model includes logistic tumor growth, immune-related tumor cell loss, a β-catenin activation term, and a field effect contribution. A baseline model without the field effect is also provided for comparison.

## Experimental data

The principal breast-cancer dataset includes 13 tumor-volume measurements at consecutive days and is loaded by `load_breast_data()` in `src/data.py`. The experiment was originally reported by Ganesh et al. and subsequently modeled by Pei et al. The stored values are treated by the current workflow as tumor volume in cm^3.

Five additional longitudinal datasets from published experimental studies are included in `src/cancer_types/`. Together with the breast-cancer case study, they provide six independently fitted tumor-growth time series. These heterogeneous datasets are used only to explore the descriptive flexibility of the model; they do not validate a universal field-cancerization mechanism or a shared parameter set across cancer types.

### Computational scales and units

The breast-cancer case study is stored as tumor volume in cm^3, converted to tumor-cell counts for model integration using `CELL_DENSITY = 1.0e7` cells/cm^3, and converted back to volume for residuals, information criteria, and plotting. The fitting objective used by the API is the manuscript MSE definition, `mean((observed - predicted)^2)`, on the same response scale used for the reported comparison.

The five non-breast exploratory datasets remain on absolute tumor-volume scales in mm^3 during model integration and plotting. They are not converted to estimated cell counts, normalized by their maxima, rescaled to `[0, 1]`, or fit with a common parameter set. The breast-cancer panel uses the principal breast-cancer case-study initialization and the explicit conversion `V = C / rho`, with `rho = 1.0e7` cells/cm^3. The Figure-4-style plots are displayed with explicit per-case metadata in `src/data.py`:

- `source_publication`
- `doi`
- `figure_panel`
- `experimental_group`
- `raw_response`
- `raw_unit`
- `transformation`
- `initial_experimental_volume_mm3`
- `model_initial_time`
- `first_observation_time`
- `first_observation_volume_mm3`
- `experimental_time_unit`
- `observed_unit`
- `model_state_unit`
- `initial_unit`
- `plot_unit`
- `observed_to_plot_scale`
- `model_to_plot_scale`

These metadata document the calculations as implemented. They also distinguish the model initial condition, the first experimental tumor-volume observation, and the corresponding experimental measurement time. The code does not reset each experimental time axis so that the first stored observation becomes `t = 0`. In the colon-cancer case, the implemented fit uses a model initial condition of approximately `7.68 mm^3` at model time `t = 0`, while the first available experimental Ctrl observation is approximately `90 mm^3` at day 10. The `7.68 mm^3` value is not an experimental measurement. These metadata should not be read as a biological assertion that the non-breast exploratory fits estimate a common biological parameter set or validate a universal FCE mechanism.

The gastric-cancer observations were digitized from relative tumor volume, `V/V0`, and are stored as absolute volume after multiplying by `V0 = 100 mm^3`. The raw relative values are preserved in `src/cancer_types/gastric_cancer/raw_relative_observations.csv`.

### Sources of the experimental datasets

- **Breast cancer:** S. Ganesh, X. Shui, K. P. Craig, J. Park, W. Wang, B. D. Brown, M. T. Abrams, "RNAi-mediated β-catenin inhibition promotes T cell infiltration and antitumor activity in combination with immune checkpoint blockade," *Molecular Therapy* 26(11) (2018) 2567–2579. [doi:10.1016/j.ymthe.2018.09.005](https://doi.org/10.1016/j.ymthe.2018.09.005). The same dataset was subsequently modeled in Y. Pei, S. Han, C. Li, J. Lei, F. Wen, "Data-based modeling of breast cancer and optimal therapy," *Journal of Theoretical Biology* 573 (2023) 111593. [doi:10.1016/j.jtbi.2023.111593](https://doi.org/10.1016/j.jtbi.2023.111593).
- **Gastric cancer:** Z. Fan, Y. Shao, X. Jiang, J. Zhou, L. Yang, H. Chen, W. Liu, "Cytotoxic effects of NIR responsive chitosan-polymersome layer coated melatonin-upconversion nanoparticles on HGC27 and AGS gastric cancer cells: Role of the ROS/PI3K/AKT/mTOR signaling pathway," *International Journal of Biological Macromolecules* 278 (2024) 134187. [doi:10.1016/j.ijbiomac.2024.134187](https://doi.org/10.1016/j.ijbiomac.2024.134187).
- **Pancreatic cancer:** X. Song, Y. Nihashi, Y. Imai, N. Mori, N. Kagaya, H. Suenaga, K. Shinya, M. Yamamoto, D. Setoyama, Y. Kunisaki, Y. S. Kida, "Collagen lattice model, populated with heterogeneous cancer-associated fibroblasts, facilitates advanced reconstruction of pancreatic cancer microenvironment," *International Journal of Molecular Sciences* 25(7) (2024) 3740. [doi:10.3390/ijms25073740](https://doi.org/10.3390/ijms25073740).
- **Colon cancer:** Y. Zhang, Y. Chen, W. Guo, Y. Guo, S. Yao, X. Wu, "Inhibition of autotaxin sensitizes colon cancer to radiation by suppressing LPAR2-AKT survival signaling," *BMC Gastroenterology* 26 (2026) 71. [doi:10.1186/s12876-025-04578-4](https://doi.org/10.1186/s12876-025-04578-4).
- **Esophageal cancer:** Z. Minghong, Y. Lin, H. Xi, X. Chunjun, S. Xiaoyi, M. Lingyun, Y. Junchang, Y. Wenyue, "Anlotinib inhibits esophageal cancer malignancy by ameliorating the immune microenvironment," *Discover Oncology* 17(1) (2026) 320. [doi:10.1007/s12672-026-04457-8](https://doi.org/10.1007/s12672-026-04457-8).
- **Skin cancer (melanoma):** H.-J. Hu, X. Liang, H.-L. Li, H.-Y. Wang, J.-F. Gu, L.-Y. Sun, J. Xiao, J.-Q. Hu, A.-M. Ni, X.-Y. Liu, "Enhanced anti-melanoma efficacy through a combination of the armed oncolytic adenovirus ZD55-IL-24 and immune checkpoint blockade in B16-bearing immunocompetent mouse model," *Cancer Immunology, Immunotherapy* 70(12) (2021) 3541–3555. [doi:10.1007/s00262-021-02946-z](https://doi.org/10.1007/s00262-021-02946-z).

## Usage

1. Install the required Python packages:
   - `numpy`
   - `pandas`
   - `scipy`
   - `matplotlib`
   - `openpyxl`
   - `nbclient` and `ipykernel`

   Install them with:

```bash
python -m pip install -r requirements.txt
```

2. Open `simulation_fce_volume_results.ipynb` and run the notebook.

3. Use the Python modules directly if needed:

```python
from src import methods

figure_3 = methods.plot_breast_comparison(candidate_count=1000)
figure_4 = methods.plot_cross_cancer()
figure_5 = methods.plot_initial_condition_scenarios()
comparison = methods.model_comparison_table()
```

4. Run the numerical verification checks:

```bash
python src/verify_analysis.py
```

5. Regenerate the Figure-5 parameter export:

```python
from src import methods
methods.export_initial_condition_parameters()
```


## Reproducibility scope

The repository is designed to reproduce the numerical outputs reported in the manuscript from the stored datasets and reported parameter values. In particular, the notebook and supporting modules allow users to:

- load the experimental datasets used in the study;
- integrate the baseline and extended model formulations;
- reproduce the main simulation figures;
- recompute residuals and the model-comparison metrics (RSS, MSE, AIC, and BIC);
- reproduce the Figure-3 Monte Carlo/admissible-region sampling with `ADMISSIBLE_MSE_EPSILON = 0.004`;
- reproduce the exploratory cross-cancer trajectories and the initial-condition scenario analysis;
- export the stored Figure-5 optimization-derived parameter values; and
- inspect the original and reparameterized model formulations.

The repository supports three distinct workflows:

- Reproduction using stored final parameter sets: the manuscript-facing figures, comparisons, and Figure-5 parameter export are generated from values stored in `src/data.py`.
- New basin-hopping optimization: `methods.fit_model()` wraps `scipy.optimize.basinhopping` with a documented fixed random seed and bounded local minimization for users who intentionally run a new fit.
- Monte Carlo/admissible-region sampling: Figure 3 samples candidate vectors and retains only those satisfying the empirical criterion `MSE(theta) <= 0.004`.

The notebook executes a lightweight deterministic fitting smoke test so that the basin-hopping path is tested, and `methods.print_fit_result()` formats any `fit_model` result using the manuscript parameter notation. However, it does **not** automatically rerun every original parameter-estimation procedure used during manuscript development. Several manuscript tables and figures are reproduced from the reported fitted parameter sets stored in `src/data.py`.

Accordingly, the repository should be interpreted as supporting reproducibility of the reported numerical results and model calculations, rather than as an archival record of every historical optimization run. Users who perform new parameter-estimation runs should record the model formulation, parameter bounds, initial guesses, initial conditions, number of basin-hopping iterations, random seed, and resulting objective-function value. The function `fit_model` in `src/methods.py` provides the fitting framework used for such analyses.

## Code, data, and materials availability

The code, data, and supporting materials used in this study are publicly available on GitHub at https://github.com/davidalejandromiranda/field_cancerization_effect_modelling and on Zenodo.

### Current release

*davidalejandromiranda/field_cancerization_effect_modelling: v2.0.0 — Supporting Information for the modeling of field cancerization of tumor volume dynamics.*

Version: v2.0.0  
Release date: August 16, 2026  
Zenodo DOI: https://doi.org/10.5281/zenodo.21967874

### Previous archived release

For traceability, the previous archived version remains available on Zenodo:

Version: v1.0.1  
Zenodo DOI: https://doi.org/10.5281/zenodo.19544133

## How to cite this repository

Please cite the current release as:

Jainer A. Gómez, Leidy J. Rojas, David A. Miranda (2026). *davidalejandromiranda/field_cancerization_effect_modelling: v2.0.0 — Supporting Information for the modeling of field cancerization of tumor volume dynamics*. Zenodo. https://doi.org/10.5281/zenodo.21967874

DOI: 10.5281/zenodo.21967874

BibTeX

```bibtex
@software{Gomez2026,
  author       = {Jainer A. Gómez and Leidy J. Rojas and David A. Miranda},
  title        = {davidalejandromiranda/field_cancerization_effect_modelling: v2.0.0 --- Supporting Information for the modeling of field cancerization of tumor volume dynamics},
  month        = aug,
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v2.0.0},
  doi          = {10.5281/zenodo.21967874},
  url          = {https://doi.org/10.5281/zenodo.21967874}
}
```

## License

This work is under the MIT license.

## Notes

- The repository supports the comparison between the field cancerization model and a baseline model without the field effect.
- Original and reparameterized formulations are included for the breast-cancer case study.
- Parameter ranges and the empirical admissibility tolerance used for Figure 3 are defined in `src/data.py`.
- Random sampling and basin hopping use a documented fixed seed to improve computational reproducibility.
- Stored fitted parameter sets are used to reproduce the manuscript tables and figures; these should not be interpreted as independent statistical validation of parameter identifiability.
- `docs/` is intentionally excluded from version control because it contains manuscript working files.
