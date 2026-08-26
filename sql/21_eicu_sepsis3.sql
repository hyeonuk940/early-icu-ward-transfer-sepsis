-- =====================================================================
-- eICU Stage B: primary Sepsis-3 cohort (suspected infection + SOFA>=2 by 24h)
--   suspected infection: ingredient-level antibiotic AND culture, both <=24h
--   first-24h SOFA (0-1440 min) from pivoted components
-- Adds sepsis3_primary flag to sepsis_work.eicu_cohort.
-- =====================================================================
-- antibiotic within first 24h (ingredient-level, administered)
DROP TABLE IF EXISTS sepsis_work.eicu_abx24;
CREATE TABLE sepsis_work.eicu_abx24 AS
SELECT DISTINCT patientunitstayid
FROM eicu_crd.medication
WHERE COALESCE(drugordercancelled,'No')='No'
  AND COALESCE(drugstartoffset, drugorderoffset) BETWEEN -1440 AND 1440
  AND drugname ~* 'vancomycin|piperacillin|tazobactam|zosyn|cefepime|ceftriaxone|rocephin|cefazolin|ancef|meropenem|merrem|metronidazole|flagyl|levofloxacin|levaquin|ciprofloxacin|cipro|azithromycin|clindamycin|gentamicin|tobramycin|amikacin|ampicillin|sulbactam|unasyn|aztreonam|ceftazidime|cefuroxime|ertapenem|imipenem|linezolid|daptomycin|doxycycline|tigecycline|nafcillin|oxacillin|penicillin|amoxicillin|clavulanate|augmentin|sulfamethoxazole|trimethoprim|bactrim|moxifloxacin|colistin|polymyxin|ceftaroline|cefotaxime|cefoxitin';

-- culture within first 24h (microlab OR treatment-string culture)
DROP TABLE IF EXISTS sepsis_work.eicu_cx24;
CREATE TABLE sepsis_work.eicu_cx24 AS
SELECT DISTINCT patientunitstayid FROM (
  SELECT patientunitstayid FROM eicu_crd.microlab WHERE culturetakenoffset BETWEEN -1440 AND 1440
  UNION
  SELECT patientunitstayid FROM eicu_crd.treatment
   WHERE treatmentoffset BETWEEN -1440 AND 1440 AND LOWER(treatmentstring) ~ 'culture'
) q;

-- first-24h SOFA components (worst value 0-1440 min)
DROP TABLE IF EXISTS sepsis_work.eicu_sofa24;
CREATE TABLE sepsis_work.eicu_sofa24 AS
WITH lab AS (
  SELECT patientunitstayid,
    MIN(platelets) AS plt_min, MAX(bilirubin) AS bili_max, MAX(creatinine) AS cr_max
  FROM eicu_derived.pivoted_lab WHERE chartoffset BETWEEN 0 AND 1440 GROUP BY 1
),
vit AS (
  SELECT patientunitstayid, MIN(nibp_mean) AS map_min
  FROM eicu_derived.pivoted_vital WHERE chartoffset BETWEEN 0 AND 1440 GROUP BY 1
),
gcs AS (
  SELECT patientunitstayid, MIN(gcs) AS gcs_min
  FROM eicu_derived.pivoted_gcs WHERE chartoffset BETWEEN 0 AND 1440 GROUP BY 1
),
med AS (
  SELECT patientunitstayid,
    MAX(GREATEST(COALESCE(norepinephrine,0),COALESCE(epinephrine,0),COALESCE(dopamine,0),
                 COALESCE(dobutamine,0),COALESCE(vasopressin,0),COALESCE(phenylephrine,0))) AS pressor
  FROM eicu_derived.pivoted_med WHERE chartoffset BETWEEN 0 AND 1440 GROUP BY 1
)
SELECT c.patientunitstayid,
  (CASE WHEN l.plt_min IS NULL THEN 0 WHEN l.plt_min<20 THEN 4 WHEN l.plt_min<50 THEN 3 WHEN l.plt_min<100 THEN 2 WHEN l.plt_min<150 THEN 1 ELSE 0 END
 + CASE WHEN l.bili_max IS NULL THEN 0 WHEN l.bili_max>=12 THEN 4 WHEN l.bili_max>=6 THEN 3 WHEN l.bili_max>=2 THEN 2 WHEN l.bili_max>=1.2 THEN 1 ELSE 0 END
 + CASE WHEN m.pressor>0 THEN 3 WHEN v.map_min IS NOT NULL AND v.map_min<70 THEN 1 ELSE 0 END
 + CASE WHEN g.gcs_min IS NULL THEN 0 WHEN g.gcs_min<6 THEN 4 WHEN g.gcs_min<=9 THEN 3 WHEN g.gcs_min<=12 THEN 2 WHEN g.gcs_min<=14 THEN 1 ELSE 0 END
 + CASE WHEN l.cr_max IS NULL THEN 0 WHEN l.cr_max>=5 THEN 4 WHEN l.cr_max>=3.5 THEN 3 WHEN l.cr_max>=2 THEN 2 WHEN l.cr_max>=1.2 THEN 1 ELSE 0 END
  ) AS sofa24
FROM sepsis_work.eicu_cohort c
LEFT JOIN lab l ON l.patientunitstayid=c.patientunitstayid
LEFT JOIN vit v ON v.patientunitstayid=c.patientunitstayid
LEFT JOIN gcs g ON g.patientunitstayid=c.patientunitstayid
LEFT JOIN med m ON m.patientunitstayid=c.patientunitstayid;

-- flag primary Sepsis-3
ALTER TABLE sepsis_work.eicu_cohort
  ADD COLUMN IF NOT EXISTS sofa24 int,
  ADD COLUMN IF NOT EXISTS sepsis3_primary int;
UPDATE sepsis_work.eicu_cohort c
SET sofa24 = s.sofa24 FROM sepsis_work.eicu_sofa24 s WHERE s.patientunitstayid=c.patientunitstayid;
UPDATE sepsis_work.eicu_cohort c
SET sepsis3_primary = CASE
  WHEN c.patientunitstayid IN (SELECT patientunitstayid FROM sepsis_work.eicu_abx24)
   AND c.patientunitstayid IN (SELECT patientunitstayid FROM sepsis_work.eicu_cx24)
   AND COALESCE(c.sofa24,0) >= 2 THEN 1 ELSE 0 END;

-- report
SELECT 'age>=18' m, count(*)::int v FROM sepsis_work.eicu_cohort WHERE age>=18
UNION ALL SELECT 'abx24', count(*)::int FROM sepsis_work.eicu_cohort c WHERE age>=18 AND c.patientunitstayid IN (SELECT patientunitstayid FROM sepsis_work.eicu_abx24)
UNION ALL SELECT 'cx24', count(*)::int FROM sepsis_work.eicu_cohort c WHERE age>=18 AND c.patientunitstayid IN (SELECT patientunitstayid FROM sepsis_work.eicu_cx24)
UNION ALL SELECT 'sepsis3_primary', count(*)::int FROM sepsis_work.eicu_cohort WHERE age>=18 AND sepsis3_primary=1
UNION ALL SELECT 'sepsis3_primary+landmark', count(*)::int FROM sepsis_work.eicu_cohort WHERE age>=18 AND sepsis3_primary=1 AND in_icu_at_24h=1
UNION ALL SELECT 'n_hosp_primary', count(distinct hospitalid)::int FROM sepsis_work.eicu_cohort WHERE age>=18 AND sepsis3_primary=1
ORDER BY m;
