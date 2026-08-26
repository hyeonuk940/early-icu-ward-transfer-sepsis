-- =====================================================================
-- Stage 2d: 56-feature extraction (first 24h) on landmark cohort
-- Ports feature CTEs from MIMIC_before_null.txt (BigQuery) to Postgres.
-- Cohort = sepsis_work.mimic_cohort WHERE in_landmark (n=22,825).
-- Missing values RETAINED (native-missing handled later in Python) — NO
-- listwise deletion, NO whole-record outlier deletion here.
-- =====================================================================
DROP TABLE IF EXISTS sepsis_work.mimic_features;
CREATE TABLE sepsis_work.mimic_features AS
WITH base AS (
  SELECT * FROM sepsis_work.mimic_cohort WHERE in_landmark
),
vaso AS (  -- vasopressor within 24h of ICU intime
  SELECT b.stay_id,
    MAX(CASE WHEN vp.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS vasopressin,
    MAX(CASE WHEN ep.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS epinephrine,
    MAX(CASE WHEN ne.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS norepinephrine
  FROM base b
  LEFT JOIN mimiciv_derived.vasopressin vp
    ON vp.stay_id=b.stay_id AND vp.starttime < b.icu_intime + INTERVAL '24 hour'
       AND COALESCE(vp.endtime, vp.starttime) >= b.icu_intime
  LEFT JOIN mimiciv_derived.epinephrine ep
    ON ep.stay_id=b.stay_id AND ep.starttime < b.icu_intime + INTERVAL '24 hour'
       AND COALESCE(ep.endtime, ep.starttime) >= b.icu_intime
  LEFT JOIN mimiciv_derived.norepinephrine ne
    ON ne.stay_id=b.stay_id AND ne.starttime < b.icu_intime + INTERVAL '24 hour'
       AND COALESCE(ne.endtime, ne.starttime) >= b.icu_intime
  GROUP BY b.stay_id
),
vent AS (  -- invasive mechanical ventilation within 24h
  SELECT b.stay_id, 1 AS invasive_mechanical_ventilation
  FROM base b
  JOIN mimiciv_derived.ventilation v
    ON v.stay_id=b.stay_id AND v.ventilation_status='InvasiveVent'
       AND v.starttime < b.icu_intime + INTERVAL '24 hour'
       AND COALESCE(v.endtime, v.starttime) >= b.icu_intime
  GROUP BY b.stay_id
),
rrt AS (  -- CRRT specifically (match original: dialysis_type contains CRRT)
  SELECT stay_id,
    MAX(CASE WHEN dialysis_type ILIKE '%CRRT%' THEN 1 ELSE 0 END) AS crrt
  FROM mimiciv_derived.first_day_rrt GROUP BY stay_id
)
SELECT
  b.subject_id, b.hadm_id, b.stay_id,
  -- outcomes
  b.primary_safe_transfer, b.secondary_transfer_72h,
  b.icu_readmit_7d, b.death_7d_post, b.left_icu_alive, b.went_to_ward,
  b.anchor_year_group, b.icu_duration_hours,
  -- demographics
  b.age,
  CASE WHEN b.gender='M' THEN 1 ELSE 0 END AS male,
  CASE WHEN b.race ILIKE 'WHITE%' THEN 1 ELSE 0 END AS race_white,
  CASE WHEN b.race ILIKE 'BLACK%' THEN 1 ELSE 0 END AS race_black,
  CASE WHEN b.race ILIKE 'HISPANIC%' THEN 1 ELSE 0 END AS race_hispanic,
  CASE WHEN b.race ILIKE 'ASIAN%' THEN 1 ELSE 0 END AS race_asian,
  CASE WHEN b.race NOT ILIKE 'WHITE%' AND b.race NOT ILIKE 'BLACK%'
        AND b.race NOT ILIKE 'HISPANIC%' AND b.race NOT ILIKE 'ASIAN%' THEN 1 ELSE 0 END AS race_others,
  CASE WHEN b.admission_type ILIKE '%EMER%' OR b.admission_type ILIKE '%URGENT%' THEN 1 ELSE 0 END AS admission_emergency,
  CASE WHEN b.admission_type ILIKE '%ELECTIVE%' OR b.admission_type ILIKE '%SURGICAL SAME DAY%' THEN 1 ELSE 0 END AS admission_elective,
  CASE WHEN b.admission_type IS NULL
        OR (b.admission_type NOT ILIKE '%EMER%' AND b.admission_type NOT ILIKE '%URGENT%'
            AND b.admission_type NOT ILIKE '%ELECTIVE%' AND b.admission_type NOT ILIKE '%SURGICAL SAME DAY%')
       THEN 1 ELSE 0 END AS admission_other,
  -- severity / comorbidity
  ch.charlson_comorbidity_index AS cci,
  a3.apsiii,
  g.gcs_min AS gcs_score,
  SAFE_BMI.bmi,
  u.urineoutput AS urine_output,
  COALESCE(rrt.crrt,0) AS crrt,
  COALESCE(vaso.vasopressin,0) AS vasopressin,
  COALESCE(vaso.epinephrine,0) AS epinephrine,
  COALESCE(vaso.norepinephrine,0) AS norepinephrine,
  COALESCE(vent.invasive_mechanical_ventilation,0) AS invasive_mechanical_ventilation,
  -- vitals min/max
  vs.heart_rate_min, vs.heart_rate_max, vs.sbp_min, vs.sbp_max,
  vs.dbp_min, vs.dbp_max, vs.mbp_min, vs.mbp_max,
  vs.resp_rate_min AS respiratory_rate_min, vs.resp_rate_max AS respiratory_rate_max,
  vs.temperature_min, vs.temperature_max, vs.spo2_min, vs.spo2_max,
  -- labs min/max
  l.bun_min, l.bun_max, l.creatinine_min, l.creatinine_max,
  l.wbc_min, l.wbc_max, l.hemoglobin_min, l.hemoglobin_max,
  l.platelets_min AS platelet_min, l.platelets_max AS platelet_max,
  l.chloride_min, l.chloride_max, l.sodium_min, l.sodium_max,
  l.bicarbonate_min, l.bicarbonate_max, l.potassium_min, l.potassium_max,
  l.glucose_min, l.glucose_max, l.calcium_min, l.calcium_max
FROM base b
LEFT JOIN mimiciv_derived.charlson ch ON ch.hadm_id=b.hadm_id
LEFT JOIN sepsis_work.apsiii a3 ON a3.stay_id=b.stay_id
LEFT JOIN mimiciv_derived.first_day_gcs g ON g.stay_id=b.stay_id
LEFT JOIN mimiciv_derived.first_day_urine_output u ON u.stay_id=b.stay_id
LEFT JOIN mimiciv_derived.first_day_vitalsign vs ON vs.stay_id=b.stay_id
LEFT JOIN mimiciv_derived.first_day_lab l ON l.stay_id=b.stay_id
LEFT JOIN LATERAL (
  SELECT CASE WHEN w.weight IS NOT NULL AND h.height IS NOT NULL AND h.height>0
              THEN w.weight/POWER(h.height/100.0,2) END AS bmi
  FROM (SELECT weight FROM mimiciv_derived.first_day_weight fw WHERE fw.stay_id=b.stay_id LIMIT 1) w
  FULL JOIN (SELECT height FROM mimiciv_derived.first_day_height fh WHERE fh.stay_id=b.stay_id LIMIT 1) h ON TRUE
) SAFE_BMI ON TRUE
LEFT JOIN rrt ON rrt.stay_id=b.stay_id
LEFT JOIN vaso ON vaso.stay_id=b.stay_id
LEFT JOIN vent ON vent.stay_id=b.stay_id;

SELECT count(*)::int n_rows, count(distinct stay_id)::int n_stays,
       round(100.0*avg(primary_safe_transfer),1) primary_pct
FROM sepsis_work.mimic_features;
