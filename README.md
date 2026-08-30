# Early ICU-to-ward transfer in sepsis — analysis code

Code and derived results for *Machine Learning Prediction of Early ICU-to-Ward Transfer
in Patients with Sepsis: Development and External Validation Using MIMIC-IV and eICU-CRD*.

A LightGBM model is developed on MIMIC-IV and validated externally on eICU-CRD to predict
**ICU-to-ward transfer between 24 and 72 hours after ICU admission, without ICU readmission
or death within 7 days**, among patients meeting Sepsis-3 criteria by 24 hours and alive and
still in the ICU at that landmark.

The repository holds the extraction SQL, the analysis code and the aggregate results the
paper reports. Results and their interpretation are in the paper; this file describes the
data, the code and how to run it.

---

## Data

**The patient-level CSVs are not in this repository and must not be redistributed.**
MIMIC-IV and eICU-CRD are credentialed PhysioNet resources whose data use agreements
prohibit it. To reproduce the analysis:

1. Complete the CITI training and obtain credentialed access at
   [physionet.org](https://physionet.org) for **MIMIC-IV v3.1** and **eICU-CRD v2.0**.
2. Load both into a local PostgreSQL instance.
3. Run the SQL in `sql/` in the order below. The final two scripts write
   `data/mimic_landmark_features_v1.csv` and `data/eicu_landmark_features_v1.csv`.

```
sql/derived_concepts/01_suspicion_of_infection.sql   antibiotic-culture pairing
sql/derived_concepts/02_sofa.sql                     hourly SOFA
sql/derived_concepts/03_apsiii.sql                   APS III
sql/derived_concepts/04_sepsis3.sql                  Sepsis-3 onset

sql/10_mimic_cohort.sql          first ICU episode, 24 h landmark, anchor-year group
sql/11_mimic_outcome.sql         readmission, death, transfer, composite outcome
sql/12_mimic_ward_transfer.sql   strict ward destination, recomputed composite
sql/13_mimic_features.sql        56 predictors  ->  data/mimic_landmark_features_v1.csv

sql/20_eicu_base_outcome.sql     cohort, outcome, APACHE diagnosis flag, APACHE score
sql/21_eicu_sepsis3.sql          antibiotic, culture, first-24 h SOFA  ->  Sepsis-3 flag
sql/22_eicu_features.sql         56 predictors, Charlson from ICD-9  ->  data/eicu_landmark_features_v1.csv
```

The SQL is written for PostgreSQL. `derived_concepts/` ports the MIT
[mimic-code](https://github.com/MIT-LCP/mimic-code) concepts; the eICU equivalents were
written for this study because no official concepts exist.

`data/DATA_DICTIONARY.md` describes the exported columns.

---

## Environment

Library versions are pinned because they change the results. scikit-learn 1.6 changed the
default AdaBoost algorithm from `SAMME.R` to `SAMME`, which moves that model's
development-CV AUC from 0.788 to 0.780; an unpinned environment will not reproduce the
published numbers.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verified on Python 3.11 with scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3,
LightGBM 4.6.0, XGBoost 3.1.3, CatBoost 1.2.8, SHAP 0.49.1 and lifelines 0.30.3.

---

## Running the analysis

Scripts resolve their own paths, so they can be run from anywhere. Within a stage the order
does not matter; between stages it does.

```bash
python src/data_prep.py            # sanity check only, writes nothing

# analysis
python src/nested_cv.py            # headline result; writes the locked threshold
python src/table1_final.py
python src/table1_overall_p.py
python src/cohort_comparison.py
python src/missingness_table.py
python src/make_table2.py
python src/threshold_metrics_ci.py
python src/delong_compare.py
python src/ablation_apsiii.py
python src/hospital_level.py
python src/dca_calibration.py
python src/subgroups.py
python src/missing_data.py
python src/horizon_competing.py
python src/shap_rfe.py
python src/eleven_algos.py
python src/supplementary_figures.py

# figures — read the CSVs written above
python src/make_roc_figures.py
python src/make_figures.py
python src/make_flowchart.py
```

`nested_cv.py` (~9 min), `eleven_algos.py` (~8 min, dominated by the SVM) and
`shap_rfe.py` (~6 min) are the slow steps; the rest finish in seconds to a minute.
A full run takes roughly 30 minutes on 16 cores.

Every script uses `random_state=42`. Confidence intervals are percentile bootstrap with
1,000 patient-level resamples.

---

## What each script does

**Shared**

| Script | |
|---|---|
| `data_prep.py` | Loads both CSVs, resolves the 56 predictors common to both databases, and applies prespecified physiologic plausibility ranges at the value level (no record is dropped). Imported by everything else. |

**Model development and evaluation**

| Script | |
|---|---|
| `nested_cv.py` | Outer 5 × inner 5 nested cross-validation on the development set, then the locked model and threshold, evaluated once per cohort. Source of the headline AUCs and of `locked_threshold.json`. |
| `make_table2.py` | Table 2 — four models × three cohorts, each at its own development-derived Youden threshold. |
| `threshold_metrics_ci.py` | Sensitivity, specificity, PPV and NPV with bootstrap CIs, plus the confusion matrices. |
| `delong_compare.py` | DeLong test of the model against the severity score alone. Prints to stdout. |
| `eleven_algos.py` | Eleven candidate algorithms under a common preprocessing pipeline, with Holm-adjusted DeLong tests. Exploratory. |
| `shap_rfe.py` | SHAP-based recursive feature elimination down to the 19-feature parsimonious model. |
| `hospital_level.py` | Per-hospital AUC in the primary external cohort, pooled by DerSimonian–Laird with a 95% prediction interval. |

**Baseline tables and sensitivity analyses**

| Script | |
|---|---|
| `table1_final.py` | Table 1 — baseline characteristics by outcome, both cohorts. |
| `table1_overall_p.py` | Variable-level p-values for the multi-category Table 1 variables; calibration intercepts for every model and cohort. |
| `cohort_comparison.py` | Development set versus each evaluation cohort, with standardised mean differences. |
| `missingness_table.py` | Variable-level missingness in all three cohorts. |
| `ablation_apsiii.py` | Does the model add anything beyond the severity score? Four predictor sets. |
| `dca_calibration.py` | Decision-curve analysis and calibration bins. |
| `subgroups.py` | Discrimination, calibration and error rates by ventilation, age, sex, race and comorbidity. |
| `missing_data.py` | Native missing handling versus complete case, MICE and missingness indicators. |
| `horizon_competing.py` | 48/72/96 h transfer horizons, and an Aalen–Johansen competing-risks analysis with death as the competing event. |
| `supplementary_figures.py` | Split sizes, parsimonious-model CIs and horizon CIs. |

**Figures**

| Script | |
|---|---|
| `make_roc_figures.py` | Figure 2 — two-panel ROC, 600 dpi, one file per external cohort. |
| `make_figures.py` | Calibration, decision curves, hospital forest plot, SHAP-RFE curve, competing-risks CIF and SHAP summary. |
| `make_flowchart.py` | Figure 1 — patient selection. Counts are hard-coded and must be updated if the cohort changes. |

---

## Outputs

`results/` holds 25 aggregate files (no patient-level data) and `figures/` holds the
published figures. Both are committed so the tables in the paper can be checked without
PhysioNet access.

| File | Contents |
|---|---|
| `headline_nested_cv.csv` | Nested-CV, internal-test and external AUCs with CIs, calibration and threshold metrics |
| `Table1.tsv` | Baseline characteristics by outcome, both cohorts |
| `table1_overall_pvalues.csv`, `calibration_intercepts.csv` | Variable-level p-values; calibration intercepts per model and cohort |
| `table2_performance.csv` | AUC, calibration slope, Brier, sensitivity, specificity, threshold |
| `threshold_metrics_ci.csv` | Sensitivity, specificity, PPV, NPV with CIs and confusion matrices |
| `cohort_comparison.tsv` | Development versus each evaluation cohort, standardised mean differences |
| `missingness_by_variable.csv` | Missingness per predictor, all three cohorts |
| `hospital_level_auc.csv` | Per-hospital n, events, event rate, AUC, calibration intercept |
| `ablation_apsiii.csv` | Severity-score ablation, four predictor sets |
| `eleven_algos.csv`, `eleven_algos_delong_holm.csv` | Candidate-algorithm comparison and Holm-adjusted DeLong tests |
| `subgroups.csv` | Subgroup discrimination, calibration and error rates |
| `missing_data_sensitivity.csv` | Missing-data strategy comparison |
| `horizon_sensitivity.csv`, `horizon_auc_ci.csv`, `cif_by_riskquartile.csv` | Transfer-horizon sensitivity and competing-risks CIF |
| `shap_rfe_curve.csv`, `parsimonious_*.csv` | Feature-elimination curve and the 19-feature model |
| `calibration_bins.csv`, `dca_netbenefit.csv` | Calibration bins and decision-curve net benefit |
| `split_sizes.csv` | Development and internal-test sizes and event counts |
| `locked_threshold.json` | Classification threshold written by `nested_cv.py` |

The fitted model itself is not distributed. The locked hyperparameters
(`learning_rate` 0.05, `n_estimators` 300, `num_leaves` 15, `min_child_samples` 50), the
locked classification threshold (0.396) and the code needed to refit it are all here.
