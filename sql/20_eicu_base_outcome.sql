-- =====================================================================
-- eICU Stage A: base cohort (first ICU stay per hospital stay) + outcome
--   + primary-cohort sepsis flag (APACHE sepsis admission dx, no culture)
-- Design mirrors MIMIC: 24h landmark, deaths retained,
-- composite safe ICU->ward transfer, readmission & death within 7d.
-- Timeline: all offsets in minutes; hosp-timeline unit intime = -hospitaladmitoffset.
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS sepsis_work;

-- 1. patient map with hospital-timeline unit in/out
DROP TABLE IF EXISTS sepsis_work.eicu_pmap;
CREATE TABLE sepsis_work.eicu_pmap AS
SELECT
  patientunitstayid, patienthealthsystemstayid, hospitalid, unitvisitnumber,
  gender,
  CASE WHEN age = '> 89' THEN 90 ELSE NULLIF(regexp_replace(age,'[^0-9]','','g'),'')::int END AS age,
  ethnicity,
  NULLIF(admissionheight,0)::float8 AS height,
  NULLIF(admissionweight,0)::float8 AS weight,
  COALESCE(hospitaladmitoffset,0) AS hospitaladmitoffset,
  unitdischargeoffset,
  unitdischargelocation, unitdischargestatus,
  hospitaldischargeoffset, hospitaldischargestatus,
  -COALESCE(hospitaladmitoffset,0)                        AS icu_intime_hosp,   -- min from hosp admit
  unitdischargeoffset - COALESCE(hospitaladmitoffset,0)   AS icu_outtime_hosp   -- min from hosp admit
FROM eicu_crd.patient
WHERE patientunitstayid IS NOT NULL AND patienthealthsystemstayid IS NOT NULL
  AND unitdischargeoffset IS NOT NULL AND unitdischargeoffset > 0;

-- 2. index (first) ICU stay per hospital stay
DROP TABLE IF EXISTS sepsis_work.eicu_index;
CREATE TABLE sepsis_work.eicu_index AS
SELECT * FROM (
  SELECT p.*,
    ROW_NUMBER() OVER (PARTITION BY patienthealthsystemstayid
                       ORDER BY icu_intime_hosp ASC, patientunitstayid ASC) AS rn
  FROM sepsis_work.eicu_pmap p
) q WHERE rn = 1;

-- 3. readmission: any later unit stay in same hospital stay starting 0-7d after index ICU out
DROP TABLE IF EXISTS sepsis_work.eicu_cohort;
CREATE TABLE sepsis_work.eicu_cohort AS
WITH readmit AS (
  SELECT i.patientunitstayid,
    MAX(CASE WHEN o.icu_intime_hosp > i.icu_outtime_hosp
             AND o.icu_intime_hosp <= i.icu_outtime_hosp + 7*1440 THEN 1 ELSE 0 END) AS icu_readmit_7d
  FROM sepsis_work.eicu_index i
  JOIN sepsis_work.eicu_pmap o
    ON o.patienthealthsystemstayid = i.patienthealthsystemstayid
   AND o.patientunitstayid <> i.patientunitstayid
  GROUP BY i.patientunitstayid
),
altdx AS (  -- primary cohort: APACHE sepsis admission diagnosis
  SELECT DISTINCT patientunitstayid, 1 AS sepsis_admitdx
  FROM eicu_crd.admissiondx WHERE admitdxpath ~* 'sepsis|septic'
),
aps AS (
  SELECT patientunitstayid, MAX(NULLIF(apachescore,-1)) AS apsiii
  FROM eicu_crd.apachepatientresult GROUP BY patientunitstayid
)
SELECT
  i.*,
  COALESCE(r.icu_readmit_7d,0) AS icu_readmit_7d,
  COALESCE(ad.sepsis_admitdx,0) AS sepsis_admitdx,
  a.apsiii,
  -- ICU duration (min) and hours
  i.unitdischargeoffset AS icu_duration_min,
  i.unitdischargeoffset/60.0 AS icu_duration_hours,
  -- left ICU alive
  CASE WHEN i.unitdischargestatus='Expired' THEN 0 ELSE 1 END AS left_icu_alive,
  -- ward destination (strict): Floor/SDU/Acute Care/Telemetry
  CASE WHEN i.unitdischargelocation IN
       ('Floor','Step-Down Unit (SDU)','Acute Care/Floor','Telemetry') THEN 1 ELSE 0 END AS went_to_ward,
  -- death within 7d of ICU exit (hospital expired within 7d after unit discharge)
  CASE WHEN i.hospitaldischargestatus='Expired'
        AND i.hospitaldischargeoffset IS NOT NULL
        AND (i.hospitaldischargeoffset - i.unitdischargeoffset) BETWEEN 0 AND 7*1440
       THEN 1 ELSE 0 END AS death_7d_post,
  -- landmark: alive & in ICU at 24h
  CASE WHEN i.unitdischargeoffset > 1440
        AND NOT (i.unitdischargestatus='Expired' AND i.unitdischargeoffset <= 1440)
       THEN 1 ELSE 0 END AS in_icu_at_24h
FROM sepsis_work.eicu_index i
LEFT JOIN readmit r ON r.patientunitstayid = i.patientunitstayid
LEFT JOIN altdx ad ON ad.patientunitstayid = i.patientunitstayid
LEFT JOIN aps a ON a.patientunitstayid = i.patientunitstayid;

-- report
SELECT 'index_icu_stays' m, count(*)::int v FROM sepsis_work.eicu_cohort
UNION ALL SELECT 'age_ge_18', count(*)::int FROM sepsis_work.eicu_cohort WHERE age>=18
UNION ALL SELECT 'alt_sepsis_admitdx', count(*)::int FROM sepsis_work.eicu_cohort WHERE age>=18 AND sepsis_admitdx=1
UNION ALL SELECT 'alt_sepsis_+landmark', count(*)::int FROM sepsis_work.eicu_cohort WHERE age>=18 AND sepsis_admitdx=1 AND in_icu_at_24h=1
UNION ALL SELECT 'n_hospitals_alt_sepsis', count(distinct hospitalid)::int FROM sepsis_work.eicu_cohort WHERE age>=18 AND sepsis_admitdx=1
ORDER BY m;
