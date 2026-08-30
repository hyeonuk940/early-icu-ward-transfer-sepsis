# Data dictionary — analysis-ready feature tables

Two CSVs, one row per patient (first ICU episode, 24-hour landmark cohort).
Predictor **names are identical** across both files (56 predictors) so a MIMIC-trained model
applies directly to eICU. Missing values are **retained** (not imputed) for native-missing modeling;
individual values outside fixed physiologic ranges are set to missing at analysis time
(see `src/data_prep.py`), never by deleting the row.

## `mimic_landmark_features_v1.csv` — 22,825 × 67 (development)
## `eicu_landmark_features_v1.csv` — 14,399 × 68 (external validation)

> Cohort unit = one first ICU episode per patient (MIMIC subject_id; eICU uniquepid).

### Identifiers
| Column | DB | Description |
|---|---|---|
| subject_id, hadm_id, stay_id | MIMIC | patient / hospital-admission / ICU-stay id |
| patientunitstayid, patienthealthsystemstayid | eICU | ICU-stay / hospital-stay id |
| hospitalid | eICU | hospital id (for site-level analysis) |

### Outcomes & bookkeeping
| Column | Description |
|---|---|
| **primary_safe_transfer** | **Primary outcome (1/0):** ICU→ward transfer 24–72 h **and** no ICU readmission or death within 7 d of transfer |
| secondary_transfer_72h | Secondary outcome (1/0): observed ICU exit ≤72 h (original definition) |
| icu_readmit_7d | ICU readmission within 7 d of index ICU exit (1/0) |
| death_7d_post | Death within 7 d of index ICU exit (1/0) |
| left_icu_alive | Left the ICU alive (1/0) |
| went_to_ward | Immediate post-ICU destination was a general/step-down ward (1/0) |
| icu_duration_hours | Index ICU episode duration (hours) |
| anchor_year_group | MIMIC only: 3-year band (temporal split) |
| **sepsis3_primary** | eICU only: **sensitivity cohort** — Sepsis-3 criteria, culture + antibiotic + first-24 h SOFA≥2 (1/0) |
| **sepsis_admitdx** | eICU only: **primary cohort** — APACHE sepsis admission diagnosis (1/0) |

> eICU external cohorts are selected by flag: **primary** = `sepsis_admitdx==1` (n=13,384, 200 hospitals);
> **sensitivity** = `sepsis3_primary==1` (n=2,409, 84 hospitals). The column names date from an
> earlier revision in which the two labels were reversed; they are kept so the CSVs still match `sql/`.

### Predictors (56; first-24h; identical names in both files)
- **Demographics/administrative:** age; male; race_white/black/hispanic/asian/others;
  admission_emergency/elective/other.
- **Severity/comorbidity:** cci (Charlson); apsiii (APS III — MIMIC true APS III; **eICU is the
  APACHE-IVa score used as a proxy**, note the scale difference); gcs_score; bmi.
- **Treatments (first 24 h):** crrt; vasopressin; epinephrine; norepinephrine;
  invasive_mechanical_ventilation.
- **Renal:** urine_output (24 h).
- **Vital signs (min & max, first 24 h):** heart_rate, sbp, dbp, mbp, respiratory_rate,
  temperature, spo2.
- **Laboratory (min & max, first 24 h):** bun, creatinine, wbc, hemoglobin, platelet, chloride,
  sodium, bicarbonate, potassium, glucose, calcium.

### Notes for re-analysis
- Non-predictor columns to exclude when modeling: the identifiers and all outcome/bookkeeping
  columns above (see `NON_FEATURES` in `src/data_prep.py`).
- Development/internal split is temporal: internal test = `anchor_year_group` in
  {2017–2019, 2020–2022}; development = earlier bands.
- eICU `apsiii` = APACHE-IVa score (mean ≈70) vs MIMIC true APS III (mean ≈51); this scale mismatch
  matters for external validation and is discussed in the report/response letter.
