# Supporting Information: Field Cancerization Modeling

This repository contains the code and data used to support the manuscript on field cancerization modeling. It is intended as supplementary material to reproduce simulations and review the model structure.

## Repository contents

- `simulation_fce_volume_results.ipynb`
  - Jupyter notebook with the main simulations and figures.
- `scr/data.py`
  - Experimental tumor volume data and model parameter definitions.
- `scr/methods.py`
  - Implementation of the field cancerization model and the baseline reference model.
- `scr/others_non_used.py`
  - Archived helper functions kept for reference.

## Model overview

The model describes tumor volume evolution with an additional field effect state variable. The main variables are:

- `C`: tumor cells.
- `N`: preconditioning normal cells.

The full model includes logistic tumor growth, immune-related tumor cell loss, a β-catenin activation term, and a field effect contribution. A baseline model without the field effect is also provided for comparison.

## Experimental data

The dataset includes 13 tumor volume measurements at consecutive days and is loaded by `load_pei_data()` in `scr/data.py`.

The values were extracted numerically from the published figure in Pei et al.:

Y. Pei, S. Han, C. Li, J. Lei, F. Wen, "Data-based modeling of breast cancer and optimal therapy," Journal of Theoretical Biology 573 (2023) 111593. [doi:10.1016/j.jtbi.2023.111593](https://doi.org/10.1016/j.jtbi.2023.111593).

URL: https://www.sciencedirect.com/science/article/pii/S002251932300190X

## Usage

1. Install the required Python packages:
   - `numpy`
   - `pandas`
   - `scipy`
   - `matplotlib`
   - `jupyter`

2. Open `simulation_fce_volume_results.ipynb` and run the notebook.

3. Use the Python modules directly if needed:

```python
from scr.methods import plot_model, plot_models, OF, OF2
from scr.data import p0, intervals
```

## Code, data, and materials availability

The code, data, and supporting materials used in this study are publicly available on GitHub at https://github.com/davidalejandromiranda/field_cancerization_effect_modelling and on Zenodo as:

*davidalejandromiranda/field_cancerization_effect_modelling: Supporting Information for the modeling of field cancerization of tumor volume dynamics.*

Zenodo DOI: https://doi.org/10.5281/zenodo.19544133 (v1.0.1)

## How to cite this repository

Jainer A. Gómez, Leidy J. Rojas, David A. Miranda (2026). davidalejandromiranda/field_cancerization_effect_modelling: Supporting Information for the modeling of field cancerization of tumor volume dynamics (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.19544133

DOI: 10.5281/zenodo.19544133

BibTeX

```bibtex
@software{Gomez2026,
  author       = {Jainer A. Gómez and Leidy J. Rojas and David A. Miranda},
  title        = {davidalejandromiranda/field_cancerization_effect_modelling: Supporting Information for the modeling of field cancerization of tumor volume dynamics},
  month        = apr,
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.19544133},
  url          = {https://doi.org/10.5281/zenodo.19544133}
}
```

License

This work is under the MIT license.

## Notes

- The repository supports the comparison between the field cancerization model and a baseline model without the field effect.
- Parameter ranges are defined in `scr/data.py` to explore model uncertainty.
