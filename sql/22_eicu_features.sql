-- =====================================================================
-- eICU Stage C: 56-feature extraction (first 24h) for landmark cohort.
-- Column names aligned to MIMIC mimic_landmark_features_v1 for pooled Python.
-- Cohort = age>=18 AND in_icu_at_24h AND (sepsis3_primary OR sepsis_admitdx).
-- Missing RETAINED. CCI ported from ICD9 (eICU_before_null.txt logic).
-- =====================================================================
DROP TABLE IF EXISTS sepsis_work.eicu_features;
CREATE TABLE sepsis_work.eicu_features AS
WITH base AS (
  SELECT * FROM sepsis_work.eicu_cohort
  WHERE age>=18 AND in_icu_at_24h=1 AND (sepsis3_primary=1 OR sepsis_admitdx=1)
),
-- ---- CCI from ICD9 (Charlson) ----
dx AS (
  SELECT DISTINCT d.patientunitstayid, (m.code)[1] AS icd
  FROM eicu_crd.diagnosis d,
       LATERAL regexp_matches(REPLACE(UPPER(d.icd9code),'.',''), '([VE]?[0-9]{3,5})','g') AS m(code)
  WHERE d.icd9code IS NOT NULL
),
com AS (
  SELECT patientunitstayid,
    MAX((substr(icd,1,3) IN ('410','412'))::int) mi,
    MAX((substr(icd,1,3)='428' OR substr(icd,1,5) IN ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') OR substr(icd,1,4) BETWEEN '4254' AND '4259')::int) chf,
    MAX((substr(icd,1,3) IN ('440','441') OR substr(icd,1,4) IN ('0930','4373','4471','5571','5579','V434') OR substr(icd,1,4) BETWEEN '4431' AND '4439')::int) pvd,
    MAX((substr(icd,1,3) BETWEEN '430' AND '438' OR substr(icd,1,5)='36234')::int) cevd,
    MAX((substr(icd,1,3)='290' OR substr(icd,1,4) IN ('2941','3312'))::int) dem,
    MAX((substr(icd,1,3) BETWEEN '490' AND '505' OR substr(icd,1,4) IN ('4168','4169','5064','5081','5088'))::int) cpd,
    MAX((substr(icd,1,3)='725' OR substr(icd,1,4) IN ('4465','7100','7101','7102','7103','7104','7140','7141','7142','7148'))::int) rheum,
    MAX((substr(icd,1,3) IN ('531','532','533','534'))::int) pud,
    MAX((substr(icd,1,3) IN ('570','571') OR substr(icd,1,4) IN ('0706','0709','5733','5734','5738','5739','V427') OR substr(icd,1,5) IN ('07022','07023','07032','07033','07044','07054'))::int) mld,
    MAX((substr(icd,1,4) IN ('2500','2501','2502','2503','2508','2509'))::int) dm,
    MAX((substr(icd,1,4) IN ('2504','2505','2506','2507'))::int) dmcc,
    MAX((substr(icd,1,3) IN ('342','343') OR substr(icd,1,4) IN ('3341','3440','3441','3442','3443','3444','3445','3446','3449'))::int) para,
    MAX((substr(icd,1,3) IN ('582','585','586','V56') OR substr(icd,1,4) IN ('5880','V420','V451') OR substr(icd,1,4) BETWEEN '5830' AND '5837' OR substr(icd,1,5) IN ('40301','40311','40391','40402','40403','40412','40413','40492','40493'))::int) renal,
    MAX((substr(icd,1,3) BETWEEN '140' AND '172' OR substr(icd,1,4) BETWEEN '1740' AND '1958' OR substr(icd,1,3) BETWEEN '200' AND '208' OR substr(icd,1,4)='2386')::int) canc,
    MAX((substr(icd,1,4) IN ('4560','4561','4562') OR substr(icd,1,4) BETWEEN '5722' AND '5728')::int) sld,
    MAX((substr(icd,1,3) IN ('196','197','198','199'))::int) mets,
    MAX((substr(icd,1,3) IN ('042','043','044'))::int) aids
  FROM dx GROUP BY patientunitstayid
),
cci AS (
  SELECT patientunitstayid,
    mi+chf+pvd+cevd+dem+cpd+rheum+pud
    + GREATEST(mld, 3*sld) + GREATEST(2*dmcc, dm) + GREATEST(2*canc, 6*mets)
    + 2*para + 2*renal + 6*aids AS cci
  FROM com
),
-- ---- first-24h features ----
vit AS (
  SELECT patientunitstayid,
    MIN(heartrate) heart_rate_min, MAX(heartrate) heart_rate_max,
    MIN(nibp_systolic) sbp_min, MAX(nibp_systolic) sbp_max,
    MIN(nibp_diastolic) dbp_min, MAX(nibp_diastolic) dbp_max,
    MIN(nibp_mean) mbp_min, MAX(nibp_mean) mbp_max,
    MIN(respiratoryrate) respiratory_rate_min, MAX(respiratoryrate) respiratory_rate_max,
    MIN(temperature) temperature_min, MAX(temperature) temperature_max,
    MIN(spo2) spo2_min, MAX(spo2) spo2_max
  FROM eicu_derived.pivoted_vital WHERE chartoffset BETWEEN 0 AND 1440 GROUP BY 1
),
lab AS (
  SELECT patientunitstayid,
    MIN(bun) bun_min, MAX(bun) bun_max, MIN(creatinine) creatinine_min, MAX(creatinine) creatinine_max,
    MIN(wbc) wbc_min, MAX(wbc) wbc_max, MIN(hemoglobin) hemoglobin_min, MAX(hemoglobin) hemoglobin_max,
    MIN(platelets) platelet_min, MAX(platelets) platelet_max,
    MIN(chloride) chloride_min, MAX(chloride) chloride_max,
    MIN(sodium) sodium_min, MAX(sodium) sodium_max,
    MIN(bicarbonate) bicarbonate_min, MAX(bicarbonate) bicarbonate_max,
    MIN(potassium) potassium_min, MAX(potassium) potassium_max,
    MIN(glucose) glucose_min, MAX(glucose) glucose_max,
    MIN(calcium) calcium_min, MAX(calcium) calcium_max
  FROM eicu_derived.pivoted_lab WHERE chartoffset BETWEEN 0 AND 1440 GROUP BY 1
),
gcs AS (SELECT patientunitstayid, MIN(gcs) gcs_score FROM eicu_derived.pivoted_gcs WHERE chartoffset BETWEEN 0 AND 1440 GROUP BY 1),
uo AS (SELECT patientunitstayid, SUM(urineoutput) urine_output FROM eicu_derived.pivoted_uo WHERE chartoffset BETWEEN 0 AND 1440 GROUP BY 1),
med AS (
  SELECT patientunitstayid,
    MAX((COALESCE(vasopressin,0)>0)::int) vasopressin,
    MAX((COALESCE(epinephrine,0)>0)::int) epinephrine,
    MAX((COALESCE(norepinephrine,0)>0)::int) norepinephrine
  FROM eicu_derived.pivoted_med WHERE chartoffset BETWEEN 0 AND 1440 GROUP BY 1
),
vent AS (
  SELECT DISTINCT patientunitstayid, 1 AS invasive_mechanical_ventilation
  FROM eicu_crd.respiratorycare
  WHERE ventstartoffset <= 1440 AND LOWER(TRIM(airwaytype)) IN ('oral ett','nasal ett','tracheostomy')
),
crrt AS (
  SELECT DISTINCT patientunitstayid, 1 AS crrt
  FROM eicu_crd.treatment
  WHERE treatmentoffset BETWEEN 0 AND 1440
    AND LOWER(treatmentstring) ~ 'c ?v ?v ?h|c ?a ?v ?h'
)
SELECT
  b.patientunitstayid, b.patienthealthsystemstayid, b.hospitalid,
  -- outcomes / cohort flags
  CASE WHEN b.icu_duration_hours<=72 AND b.left_icu_alive=1 AND b.went_to_ward=1
        AND b.icu_readmit_7d=0 AND b.death_7d_post=0 THEN 1 ELSE 0 END AS primary_safe_transfer,
  CASE WHEN b.icu_duration_hours<=72 THEN 1 ELSE 0 END AS secondary_transfer_72h,
  b.sepsis3_primary, b.sepsis_admitdx, b.icu_readmit_7d, b.death_7d_post,
  b.left_icu_alive, b.went_to_ward, b.icu_duration_hours,
  -- demographics
  b.age,
  CASE WHEN b.gender='Male' THEN 1 ELSE 0 END AS male,
  CASE WHEN b.ethnicity='Caucasian' THEN 1 ELSE 0 END AS race_white,
  CASE WHEN b.ethnicity='African American' THEN 1 ELSE 0 END AS race_black,
  CASE WHEN b.ethnicity='Hispanic' THEN 1 ELSE 0 END AS race_hispanic,
  CASE WHEN b.ethnicity='Asian' THEN 1 ELSE 0 END AS race_asian,
  CASE WHEN b.ethnicity NOT IN ('Caucasian','African American','Hispanic','Asian') OR b.ethnicity IS NULL THEN 1 ELSE 0 END AS race_others,
  CASE WHEN b.icu_intime_hosp IS NOT NULL THEN NULL END AS _pad,   -- placeholder removed below
  CASE WHEN bh.hospitaladmitsource='Emergency Department' THEN 1 ELSE 0 END AS admission_emergency,
  CASE WHEN bh.hospitaladmitsource IN ('Operating Room','Recovery Room','PACU') THEN 1 ELSE 0 END AS admission_elective,
  CASE WHEN bh.hospitaladmitsource IS NULL
        OR bh.hospitaladmitsource NOT IN ('Emergency Department','Operating Room','Recovery Room','PACU') THEN 1 ELSE 0 END AS admission_other,
  -- severity/comorbidity/treatments
  cci.cci,
  b.apsiii,
  gcs.gcs_score,
  CASE WHEN b.height IS NOT NULL AND b.weight IS NOT NULL AND b.height>0
       THEN b.weight/POWER(b.height/100.0,2) END AS bmi,
  uo.urine_output,
  COALESCE(crrt.crrt,0) AS crrt,
  COALESCE(med.vasopressin,0) AS vasopressin,
  COALESCE(med.epinephrine,0) AS epinephrine,
  COALESCE(med.norepinephrine,0) AS norepinephrine,
  COALESCE(vent.invasive_mechanical_ventilation,0) AS invasive_mechanical_ventilation,
  -- vitals
  vit.heart_rate_min, vit.heart_rate_max, vit.sbp_min, vit.sbp_max,
  vit.dbp_min, vit.dbp_max, vit.mbp_min, vit.mbp_max,
  vit.respiratory_rate_min, vit.respiratory_rate_max,
  vit.temperature_min, vit.temperature_max, vit.spo2_min, vit.spo2_max,
  -- labs
  lab.bun_min, lab.bun_max, lab.creatinine_min, lab.creatinine_max,
  lab.wbc_min, lab.wbc_max, lab.hemoglobin_min, lab.hemoglobin_max,
  lab.platelet_min, lab.platelet_max, lab.chloride_min, lab.chloride_max,
  lab.sodium_min, lab.sodium_max, lab.bicarbonate_min, lab.bicarbonate_max,
  lab.potassium_min, lab.potassium_max, lab.glucose_min, lab.glucose_max,
  lab.calcium_min, lab.calcium_max
FROM base b
LEFT JOIN eicu_crd.patient bh ON bh.patientunitstayid=b.patientunitstayid
LEFT JOIN cci ON cci.patientunitstayid=b.patientunitstayid
LEFT JOIN vit ON vit.patientunitstayid=b.patientunitstayid
LEFT JOIN lab ON lab.patientunitstayid=b.patientunitstayid
LEFT JOIN gcs ON gcs.patientunitstayid=b.patientunitstayid
LEFT JOIN uo ON uo.patientunitstayid=b.patientunitstayid
LEFT JOIN med ON med.patientunitstayid=b.patientunitstayid
LEFT JOIN vent ON vent.patientunitstayid=b.patientunitstayid
LEFT JOIN crrt ON crrt.patientunitstayid=b.patientunitstayid;

ALTER TABLE sepsis_work.eicu_features DROP COLUMN _pad;

SELECT count(*)::int n_rows,
  sum(sepsis3_primary)::int prim, sum(sepsis_admitdx)::int alt,
  round(100.0*avg(primary_safe_transfer),1) prim_outcome_pct
FROM sepsis_work.eicu_features;
