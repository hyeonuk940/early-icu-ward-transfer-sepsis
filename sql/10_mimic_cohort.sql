-- =====================================================================
-- Stage 2: MIMIC-IV cohort (24h landmark, deaths retained)
-- Builds sepsis_work.mimic_cohort with stepwise audit counts.
-- Design changes vs original (BigQuery MIMIC_before_null.txt):
--   * KEEP in-hospital deaths (no hospital_expire_flag filter)
--   * sepsis by 24h: suspected_infection_time <= intime+24h
--   * explicit 24h landmark risk set (alive & in ICU at 24h)
--   * add anchor_year_group (temporal split)
--   * carry death/transfer/readmission fields for composite outcome
-- Outcome columns are derived in a later step; here we fix the risk set.
-- =====================================================================

-- ---- 1. ICU episodes (merge stays <=6h apart), first episode only --------
DROP TABLE IF EXISTS sepsis_work.mimic_first_episode;
CREATE TABLE sepsis_work.mimic_first_episode AS
WITH icu_stays AS (
  SELECT subject_id, hadm_id, stay_id, intime, outtime
  FROM mimiciv_icu.icustays
  WHERE hadm_id IS NOT NULL AND intime IS NOT NULL AND outtime IS NOT NULL
    AND outtime > intime
),
ordered AS (
  SELECT *, LAG(outtime) OVER (PARTITION BY hadm_id ORDER BY intime) AS prev_outtime
  FROM icu_stays
),
flag AS (
  SELECT *,
    CASE WHEN prev_outtime IS NULL THEN 1
         WHEN EXTRACT(EPOCH FROM (intime - prev_outtime))/3600.0 > 6 THEN 1
         ELSE 0 END AS new_episode
  FROM ordered
),
epid AS (
  SELECT *,
    SUM(new_episode) OVER (PARTITION BY hadm_id ORDER BY intime
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS icu_episode_id
  FROM flag
),
agg AS (
  SELECT subject_id, hadm_id, icu_episode_id,
    (ARRAY_AGG(stay_id ORDER BY intime))[1] AS stay_id,
    MIN(intime) AS icu_intime,
    MAX(outtime) AS icu_outtime
  FROM epid
  GROUP BY subject_id, hadm_id, icu_episode_id
)
SELECT * FROM agg WHERE icu_episode_id = 1;

-- ---- 2. attach demographics, death, sepsis timing, anchor_year_group -----
DROP TABLE IF EXISTS sepsis_work.mimic_cohort;
CREATE TABLE sepsis_work.mimic_cohort AS
SELECT
  fe.subject_id, fe.hadm_id, fe.stay_id,
  fe.icu_intime, fe.icu_outtime,
  EXTRACT(EPOCH FROM (fe.icu_outtime - fe.icu_intime))/3600.0 AS icu_duration_hours,
  a.admittime, a.dischtime, a.deathtime, a.admission_type,
  a.discharge_location, a.hospital_expire_flag, a.race,
  p.gender, p.anchor_age AS age, p.anchor_year_group, p.dod,
  s.suspected_infection_time, s.sofa_time, s.sofa_score,
  -- landmark: sepsis criteria met by 24h after ICU intime
  (s.suspected_infection_time IS NOT NULL
     AND s.suspected_infection_time <= fe.icu_intime + INTERVAL '24 hour') AS sepsis_by_24h,
  -- alive & still in ICU at the 24h landmark
  (fe.icu_outtime > fe.icu_intime + INTERVAL '24 hour'
     AND (a.deathtime IS NULL OR a.deathtime > fe.icu_intime + INTERVAL '24 hour')) AS in_icu_at_24h
FROM sepsis_work.mimic_first_episode fe
JOIN mimiciv_hosp.admissions a ON fe.hadm_id = a.hadm_id
LEFT JOIN mimiciv_hosp.patients p ON fe.subject_id = p.subject_id
LEFT JOIN sepsis_work.sepsis3 s ON fe.stay_id = s.stay_id;

-- ---- 3. stepwise selection audit ----------------------------------------
SELECT '0_all_first_icu_episodes' AS step, count(*)::int n FROM sepsis_work.mimic_cohort
UNION ALL SELECT '1_age_ge_18', count(*)::int FROM sepsis_work.mimic_cohort WHERE age >= 18
UNION ALL SELECT '2_sepsis_by_24h', count(*)::int FROM sepsis_work.mimic_cohort WHERE age>=18 AND sepsis_by_24h
UNION ALL SELECT '3_landmark_in_icu_alive_at_24h', count(*)::int FROM sepsis_work.mimic_cohort WHERE age>=18 AND sepsis_by_24h AND in_icu_at_24h
ORDER BY step;
