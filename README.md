# Kepler photometric-variability comparison

This repository contains a minimal subset of the analysis code used in the
paper. It is provided primarily to document the implementation of the main
methods and to make the scientific workflow recoverable, rather than as a
fully automated end-to-end reproduction pipeline.

The analysis compares rotational-modulation amplitudes in Kepler Objects of
Interest (KOIs) with those in a Kepler control sample while accounting for
planet detectability and differences in stellar properties.

## Scope

The public analysis subset consists of selected files under:

- data_m15/
- pdet/
- resampling/
- likelihood_wkic_common/
- likelihood_wkic_joint/

Older analysis directories, validation and Rossby-number runs, the LAMOST
analysis, large intermediate products, trained models, posterior chains, and
the manuscript source are not part of this subset.

## Workflow

~~~text
prepared M15 KIC and KOI catalogs
                |
                v
     planet-detectability weights
          /                 \
         v                   v
 weighted resampling       weighted KIC flow
         |                        |
         v                        v
 resampling results     alpha(Teff) inference/results
~~~

## Prepared starting data

- data_m15/m15_kic.csv: prepared KIC rotational-modulation sample.
- data_m15/m15_koi.csv: prepared KOI rotational-modulation sample.

These tables are the starting point of the released workflow. Their upstream
construction from the original published catalogs is not included.

## Mapping to the paper

| Code | Paper section | Role |
|---|---|---|
| pdet/pdetutils.py and pdet/pdet_given_TR_sim.ipynb | Sections 3 and 4 | Calculate the detectability of a transiting planet for the KIC sample. |
| pdet/pdet_given_TR_sim_p-cut.ipynb | Section 5 | Calculate period-specific detectability weights. |
| resampling/resample.py and resampling/resample_given_Teff.ipynb | Section 4 | Perform detectability-weighted matching and resampling. |
| resampling/resample_given_Teff_plot_dist.ipynb | Sections 2 and 4 | Check the matched distributions and assemble the resampling diagnostics. |
| likelihood_wkic_joint/train_kic_flow.ipynb | Section 5 | Train the weighted KIC normalizing-flow reference density. |
| likelihood_wkic_joint/alpha_inference.py and likelihood_wkic_joint/run_alpha_inference.ipynb | Section 5 | Infer the temperature-dependent amplitude shift alpha(Teff). |
| likelihood_wkic_joint/comparison_alpha_t.ipynb | Section 5 | Assemble the main likelihood-analysis results. |

Shared flow training and diagnostic functions are in
likelihood_wkic_common/.

## Detectability calculation

The pdet notebooks use the prepared KIC catalog together with a simulated
planet population and stellar radii. They write the all-period and
period-specific KIC tables used by the downstream analyses:

- pdet/m15_kic_w_pdet_TR.csv
- pdet/m15_kic_w_pdet_TR_p-short.csv
- pdet/m15_kic_w_pdet_TR_p-long.csv

Run these notebooks from pdet/ so that their relative paths resolve.

## Resampling analysis

The resampling notebooks combine the prepared KOI catalog with the
detectability-weighted KIC table. They match the relevant stellar-property
distributions and compare the resulting modulation amplitudes.

The included outputs are:

- resampling/resample_m15.png
- resampling/plots_dist/resample_corner.png
- resampling/plots_dist/ecdf_by_teff_logr.png
- resampling/plots_dist/ecdf_by_teff_kepmag_no-resample.png

Run these notebooks from resampling/.

## Likelihood analysis

The likelihood analysis trains a detectability-weighted joint KIC density and
uses it as the reference distribution for the KOI amplitude-shift inference.

The paper analysis uses four temperature-based configurations:

- m15_kic_pdet_teff2_all
- m15_kic_pdet_teff5_all
- m15_kic_pdet_teff5_pshort
- m15_kic_pdet_teff5_plong

For each configuration, the selected inference files retain the posterior
summary, inference grid, run specification, and convergence diagnostics.
Trained flows and full posterior samples are not included.

The included result figures are under likelihood_wkic_joint/figs/.

## External inputs and execution notes

Some notebooks refer to inputs outside the released subset:

- data/physical_catalog.csv
- data/DR2PapTable1.txt
- data/koi_candidates.csv

The prepared detectability tables allow the later stages to be understood
without regenerating the full detectability calculation.

The detectability and resampling notebooks contain unseeded random draws, so a
fresh run is not expected to reproduce the checked-in outputs bit for bit.
Some plotting cells require a working LaTeX installation.

The likelihood notebooks can be read as records of the implementation, but
they are not expected to run from start to finish without the omitted trained
flows and posterior products.

## Software

The main likelihood analysis used FlowJAX 17.2.1 with Python 3.12, JAX 0.6.2,
Equinox 0.13.1, NumPyro 0.19.0, and ArviZ 0.21.0. Other dependencies are
visible from the imports in the released scripts and notebooks.
