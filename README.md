# Supporting Information: Field Cancerization Modeling

This repository contains the code and data used to support the manuscript on field cancerization modeling. It is intended as supplementary material to reproduce simulations and review the model structure.

## Repository contents

- `simulation_fce_volume_results.ipynb`
  - Jupyter notebook with the main simulations and figures.
- `scr/data.py`
  - Experimental tumor volume data and model parameter definitions.
- `scr/methods.py`
  - Model equations, numerical integration, fitting, diagnostics, and figures.
- `scr/tipos de cancer/`
  - Tabular observations for the six exploratory tumor-growth datasets.
- `scripts/build_notebook.py`
  - Rebuilds a clean, output-free copy of the notebook.
- `requirements.txt`
  - Reproducible Python dependency ranges.

## Model overview

The model describes tumor volume evolution with an additional field effect state variable. The main variables are:

- `C`: tumor cells.
- `N`: preconditioning normal cells.

The full model includes logistic tumor growth, immune-related tumor cell loss, a β-catenin activation term, and a field effect contribution. A baseline model without the field effect is also provided for comparison.

## Experimental data

The principal breast-cancer dataset includes 13 tumor-volume measurements at consecutive days and is loaded by `load_breast_data()` in `scr/data.py`. The experiment was originally reported by Ganesh et al. and subsequently modeled by Pei et al. The values used here were extracted numerically from the published graphical data.

Five additional longitudinal datasets from published rodent studies are included in `scr/tipos de cancer/`. Together with the breast-cancer case study, they provide six independently fitted tumor-growth time series. The numerical observations were digitized from the corresponding published figures with WebPlotDigitizer. These heterogeneous datasets are used only to explore the descriptive flexibility of the model; they do not validate a universal field-cancerization mechanism or a shared parameter set across cancer types.

### Sources of the experimental datasets

- **Breast cancer:** S. Ganesh, X. Shui, K. P. Craig, J. Park, W. Wang, B. D. Brown, M. T. Abrams, "RNAi-mediated β-catenin inhibition promotes T cell infiltration and antitumor activity in combination with immune checkpoint blockade," *Molecular Therapy* 26(11) (2018) 2567–2579. [doi:10.1016/j.ymthe.2018.09.005](https://doi.org/10.1016/j.ymthe.2018.09.005). The same dataset was subsequently modeled in Y. Pei, S. Han, C. Li, J. Lei, F. Wen, "Data-based modeling of breast cancer and optimal therapy," *Journal of Theoretical Biology* 573 (2023) 111593. [doi:10.1016/j.jtbi.2023.111593](https://doi.org/10.1016/j.jtbi.2023.111593).
- **Gastric cancer:** Z. Fan, Y. Shao, X. Jiang, J. Zhou, L. Yang, H. Chen, W. Liu, "Cytotoxic effects of NIR responsive chitosan-polymersome layer coated melatonin-upconversion nanoparticles on HGC27 and AGS gastric cancer cells: Role of the ROS/PI3K/AKT/mTOR signaling pathway," *International Journal of Biological Macromolecules* 278 (2024) 134187. [doi:10.1016/j.ijbiomac.2024.134187](https://doi.org/10.1016/j.ijbiomac.2024.134187).
- **Pancreatic cancer:** X. Song, Y. Nihashi, Y. Imai, N. Mori, N. Kagaya, H. Suenaga, K. Shinya, M. Yamamoto, D. Setoyama, Y. Kunisaki, Y. S. Kida, "Collagen lattice model, populated with heterogeneous cancer-associated fibroblasts, facilitates advanced reconstruction of pancreatic cancer microenvironment," *International Journal of Molecular Sciences* 25(7) (2024) 3740. [doi:10.3390/ijms25073740](https://doi.org/10.3390/ijms25073740).
- **Colon cancer:** Y. Zhang, Y. Chen, W. Guo, Y. Guo, S. Yao, X. Wu, "Inhibition of autotaxin sensitizes colon cancer to radiation by suppressing LPAR2-AKT survival signaling," *BMC Gastroenterology* 26(1) (2025). [doi:10.1186/s12876-025-04578-4](https://doi.org/10.1186/s12876-025-04578-4).
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
- Random sampling and basin hopping use a documented fixed seed.
- `docs/` is intentionally excluded from version control because it contains
  manuscript working files.
