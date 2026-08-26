-- =====================================================================
-- Stage 2b: Outcomes on the 24h-landmark risk set (sepsis_work.mimic_cohort)
--   secondary_transfer_72h  : observed ICU exit within 72h (original def)   [comparability]
--   icu_readmit_7d          : another ICU episode 0-7d after index outtime
--   death_7d_post_transfer  : death within 7d of index ICU outtime
--   primary_safe_transfer   : left ICU alive to ward within 24-72h AND no readmit/death <=7d
-- Restrict to the landmark cohort (age>=18, sepsis_by_24h, in_icu_at_24h).
-- =====================================================================
ALTER TABLE sepsis_work.mimic_cohort
  ADD COLUMN IF NOT EXISTS in_landmark boolean,
  ADD COLUMN IF NOT EXISTS icu_readmit_7d int,
  ADD COLUMN IF NOT EXISTS death_7d_post int,
  ADD COLUMN IF NOT EXISTS left_icu_alive int,
  ADD COLUMN IF NOT EXISTS secondary_transfer_72h int,
  ADD COLUMN IF NOT EXISTS primary_safe_transfer int;

UPDATE sepsis_work.mimic_cohort c
SET in_landmark = (c.age >= 18 AND c.sepsis_by_24h AND c.in_icu_at_24h);

-- next ICU episode within the SAME hadm, starting after index outtime
WITH nexticu AS (
  SELECT c.stay_id,
         MIN(i.intime) AS next_intime
  FROM sepsis_work.mimic_cohort c
  JOIN mimiciv_icu.icustays i
    ON i.hadm_id = c.hadm_id
   AND i.intime > c.icu_outtime
  GROUP BY c.stay_id
)
UPDATE sepsis_work.mimic_cohort c
SET icu_readmit_7d = CASE
      WHEN n.next_intime IS NOT NULL
       AND n.next_intime <= c.icu_outtime + INTERVAL '7 day' THEN 1 ELSE 0 END
FROM nexticu n
WHERE c.stay_id = n.stay_id;

UPDATE sepsis_work.mimic_cohort c
SET icu_readmit_7d = COALESCE(icu_readmit_7d, 0);

-- death within 7d of index ICU outtime (deathtime or dod)
UPDATE sepsis_work.mimic_cohort c
SET death_7d_post = CASE
  WHEN COALESCE(c.deathtime, (c.dod)::timestamp) IS NOT NULL
   AND COALESCE(c.deathtime, (c.dod)::timestamp) >= c.icu_outtime
   AND COALESCE(c.deathtime, (c.dod)::timestamp) <= c.icu_outtime + INTERVAL '7 day'
  THEN 1 ELSE 0 END;

-- left ICU alive (did not die at/before ICU exit)
UPDATE sepsis_work.mimic_cohort c
SET left_icu_alive = CASE
  WHEN c.deathtime IS NULL OR c.deathtime > c.icu_outtime THEN 1 ELSE 0 END;

-- secondary outcome: observed ICU exit (episode) within 72h of ICU intime
UPDATE sepsis_work.mimic_cohort c
SET secondary_transfer_72h = CASE
  WHEN c.icu_duration_hours <= 72 THEN 1 ELSE 0 END;

-- primary composite: safe transfer 24-72h
UPDATE sepsis_work.mimic_cohort c
SET primary_safe_transfer = CASE
  WHEN c.icu_duration_hours <= 72      -- exited ICU by 72h (>24h guaranteed by landmark)
   AND c.left_icu_alive = 1            -- left ICU alive (to ward)
   AND c.icu_readmit_7d = 0            -- no ICU readmission within 7d
   AND c.death_7d_post = 0             -- no death within 7d
  THEN 1 ELSE 0 END;

-- ---- report on the landmark cohort --------------------------------------
SELECT 'landmark_n' AS metric, count(*)::int v FROM sepsis_work.mimic_cohort WHERE in_landmark
UNION ALL SELECT 'left_icu_alive', sum(left_icu_alive)::int FROM sepsis_work.mimic_cohort WHERE in_landmark
UNION ALL SELECT 'secondary_transfer_72h', sum(secondary_transfer_72h)::int FROM sepsis_work.mimic_cohort WHERE in_landmark
UNION ALL SELECT 'icu_readmit_7d', sum(icu_readmit_7d)::int FROM sepsis_work.mimic_cohort WHERE in_landmark
UNION ALL SELECT 'death_7d_post', sum(death_7d_post)::int FROM sepsis_work.mimic_cohort WHERE in_landmark
UNION ALL SELECT 'primary_safe_transfer', sum(primary_safe_transfer)::int FROM sepsis_work.mimic_cohort WHERE in_landmark
ORDER BY metric;
