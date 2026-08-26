-- =====================================================================
-- Stage 2c: strict ICU->WARD destination (user: only ward transfer = event)
-- Immediate next non-ICU location after index ICU episode within same hadm.
--   went_to_ward = 1 iff that destination is a general/intermediate WARD
--   (not discharge, PACU, ED, observation, unknown, ICU-readmit).
-- Recompute primary_safe_transfer to require went_to_ward=1.
-- =====================================================================
ALTER TABLE sepsis_work.mimic_cohort
  ADD COLUMN IF NOT EXISTS dest_careunit text,
  ADD COLUMN IF NOT EXISTS dest_eventtype text,
  ADD COLUMN IF NOT EXISTS went_to_ward int;

-- classify careunit helper via CASE (ICU / discharge / transient / ward)
-- immediate next stint after ICU exit (intime >= icu_outtime - 1h tolerance),
-- excluding rows that are themselves ICU units (continuation)
WITH cand AS (
  SELECT c.stay_id, t.careunit, t.eventtype, t.intime,
    CASE
      WHEN t.careunit ILIKE '%Intensive Care Unit%'
        OR t.careunit IN ('Coronary Care Unit (CCU)','Trauma SICU (TSICU)') THEN 'ICU'
      WHEN t.eventtype='discharge' OR t.careunit='Discharge Lounge' THEN 'DISCHARGE'
      WHEN t.careunit IN ('PACU','UNKNOWN','Unknown','Observation',
                          'Emergency Department Observation','Emergency Department',
                          'Nursery','Special Care Nursery (SCN)') THEN 'OTHER'
      ELSE 'WARD'
    END AS unit_class
  FROM sepsis_work.mimic_cohort c
  JOIN mimiciv_hosp.transfers t
    ON t.hadm_id = c.hadm_id
   AND t.intime >= c.icu_outtime - INTERVAL '1 hour'
),
-- first stint after ICU that is NOT a continuation ICU row
ranked AS (
  SELECT stay_id, careunit, eventtype, unit_class, intime,
    ROW_NUMBER() OVER (PARTITION BY stay_id
      ORDER BY (CASE WHEN unit_class='ICU' THEN 1 ELSE 0 END), intime) AS rn
  FROM cand
  WHERE unit_class <> 'ICU'   -- destination is the first NON-ICU location
)
UPDATE sepsis_work.mimic_cohort c
SET dest_careunit = r.careunit,
    dest_eventtype = r.eventtype,
    went_to_ward = CASE WHEN r.unit_class='WARD' THEN 1 ELSE 0 END
FROM ranked r
WHERE c.stay_id = r.stay_id AND r.rn = 1;

UPDATE sepsis_work.mimic_cohort SET went_to_ward = COALESCE(went_to_ward, 0);

-- recompute strict primary composite
UPDATE sepsis_work.mimic_cohort c
SET primary_safe_transfer = CASE
  WHEN c.icu_duration_hours <= 72
   AND c.left_icu_alive = 1
   AND c.went_to_ward = 1          -- STRICT: must go to ward
   AND c.icu_readmit_7d = 0
   AND c.death_7d_post = 0
  THEN 1 ELSE 0 END;

-- ---- report -------------------------------------------------------------
SELECT 'landmark_n' m, count(*)::int v FROM sepsis_work.mimic_cohort WHERE in_landmark
UNION ALL SELECT 'left_icu_alive', sum(left_icu_alive)::int FROM sepsis_work.mimic_cohort WHERE in_landmark
UNION ALL SELECT 'secondary_transfer_72h', sum(secondary_transfer_72h)::int FROM sepsis_work.mimic_cohort WHERE in_landmark
UNION ALL SELECT 'went_to_ward(any time)', sum(went_to_ward)::int FROM sepsis_work.mimic_cohort WHERE in_landmark
UNION ALL SELECT 'primary_strict_ward_composite', sum(primary_safe_transfer)::int FROM sepsis_work.mimic_cohort WHERE in_landmark
ORDER BY m;

-- destination breakdown among those who exited ICU alive within 72h
SELECT COALESCE(
  CASE
    WHEN dest_eventtype='discharge' OR dest_careunit='Discharge Lounge' THEN 'DISCHARGE(home/hospice/SNF...)'
    WHEN went_to_ward=1 THEN 'WARD'
    ELSE 'OTHER/PACU/ED/UNKNOWN' END, 'NONE') AS destination,
  count(*)::int n
FROM sepsis_work.mimic_cohort
WHERE in_landmark AND left_icu_alive=1 AND icu_duration_hours<=72
GROUP BY 1 ORDER BY 2 DESC;
