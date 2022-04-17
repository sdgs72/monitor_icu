-- total unique admission record is 22049
-- AGE SCORE we see duplicates but is fine
-- duplicate entries take maximum when processing
-- count is 23620
drop table if exists sapsii_age_score;
create TABLE sapsii_age_score (
    select t.*, case
      when age is null then null
      when age <  40 then 0
      when age <  60 then 7
      when age <  70 then 12
      when age <  75 then 15
      when age <  80 then 16
      when age >= 80 then 18
    end as age_score 
    from (
        select adm.subject_id, adm.hadm_id , DATEDIFF(adm.intime, pat.dob)/365 as age
        FROM ADMISSIONS_FULL_METAVISION adm
        inner join patients pat
        on adm.subject_id = pat.subject_id
    ) t
);

-- Vitals score containing heart rate, blood pressure and body temperature
-- hr_score, sysbp_score, temp_score
-- 22791
drop table if exists sapsii_vitals_score;
create TABLE sapsii_vitals_score (
    select v.hadm_id, v.icustay_id
    , case
        when heartrate_max is null then null
        when heartrate_min <   40 then 11
        when heartrate_max >= 160 then 7
        when heartrate_max >= 120 then 4
        when heartrate_min  <  70 then 2
        when  heartrate_max >= 70 and heartrate_max < 120
            and heartrate_min >= 70 and heartrate_min < 120
        then 0
        end as hr_score
    , case
        when  sysbp_min is null then null
        when  sysbp_min <   70 then 13
        when  sysbp_min <  100 then 5
        when  sysbp_max >= 200 then 2
        when  sysbp_max >= 100 and sysbp_max < 200
            and sysbp_min >= 100 and sysbp_min < 200
            then 0
        end as sysbp_score
    , case
        when tempc_max is null then null
        when tempc_min <  39.0 then 0
        when tempc_max >= 39.0 then 3
        end as temp_score
    from (    
        select adm.hadm_id, adm.icustay_id, vital.heartrate_max, vital.heartrate_min, vital.sysbp_max, vital.sysbp_min, vital.tempc_max, vital.tempc_min
        FROM ADMISSIONS_FULL_METAVISION adm inner join vitals_first_day vital on adm.icustay_id = vital.icustay_id
    ) v
);




-- pao2fio2_score 
-- averaage pao2fio2_vent_min=236.895564179215811147 
-- average pao2fio2_score = 6
-- count sapsii_paf_score 21381
-- count paf 21381
-- select count(*) from sapsii_paf_score where HADM_ID is not null; --> 7941
drop table if exists cpap;
create TABLE cpap (
    select ie.icustay_id
    , min(DATE_SUB(charttime, INTERVAL 1 HOUR)) as starttime
    , max(DATE_ADD(charttime, INTERVAL 4 HOUR)) as endtime
    , max(CASE
          WHEN lower(ce.value) LIKE '%cpap%' THEN 1
          WHEN lower(ce.value) LIKE '%bipap mask%' THEN 1
        else 0 end) as cpap
  FROM ADMISSIONS_FULL_METAVISION ie
  inner join chartevents ce
    on ie.icustay_id = ce.icustay_id
    and ce.charttime between ie.intime and DATE_ADD(ie.intime, INTERVAL 1 DAY)
  where itemid in
  (
    467, 469, 226732
  )
  and (lower(ce.value) LIKE '%cpap%' or lower(ce.value) LIKE '%bipap mask%')
  AND (ce.error IS NULL OR ce.error = 0)
  group by ie.icustay_id
);

drop table if exists paf;
create table paf (
    select icustay_id
    , min(pao2fio2) as pao2fio2_vent_min
    from
    (
        select bg.icustay_id, bg.charttime
        , pao2fio2
        , case when vd.icustay_id is not null then 1 else 0 end as vent
        , case when cp.icustay_id is not null then 1 else 0 end as cpap
        from blood_gas_arterial_first_day bg
        left join ventilation_durations vd
            on bg.icustay_id = vd.icustay_id
            and bg.charttime >= vd.starttime
            and bg.charttime <= vd.endtime
        left join cpap cp
            on bg.icustay_id = cp.icustay_id
            and bg.charttime >= cp.starttime
            and bg.charttime <= cp.endtime
    ) paf1
    where vent = 1 or cpap = 1
    group by icustay_id
)


drop table if exists sapsii_paf_score;
create table sapsii_paf_score (
    select adm.HADM_ID, paf.icustay_id, case
      when pao2fio2_vent_min is null then null
      when pao2fio2_vent_min <  100 then 11
      when pao2fio2_vent_min <  200 then 9
      when pao2fio2_vent_min >= 200 then 6
        end as pao2fio2_score
    from paf inner join ADMISSIONS_FULL_METAVISION adm on paf.icustay_id=adm.icustay_id
)

-- labs score
-- 4 metrics
-- 23620
drop table if exists sapsii_lab_score;
create table sapsii_lab_score (
    select gg.hadm_id, gg.icustay_id
    , case
        when wbc_max is null then null
        when wbc_min <   1.0 then 12
        when wbc_max >= 20.0 then 3
        when wbc_max >=  1.0 and wbc_max < 20.0
        and wbc_min >=  1.0 and wbc_min < 20.0
            then 0
        end as wbc_score

    , case
        when potassium_max is null then null
        when potassium_min <  3.0 then 3
        when potassium_max >= 5.0 then 3
        when potassium_max >= 3.0 and potassium_max < 5.0
        and potassium_min >= 3.0 and potassium_min < 5.0
            then 0
        end as potassium_score
    , case
        when sodium_max is null then null
        when sodium_min  < 125 then 5
        when sodium_max >= 145 then 1
        when sodium_max >= 125 and sodium_max < 145
        and sodium_min >= 125 and sodium_min < 145
            then 0
        end as sodium_score
    , case
        when bicarbonate_max is null then null
        when bicarbonate_min <  15.0 then 5
        when bicarbonate_min <  20.0 then 3
        when bicarbonate_max >= 20.0
        and bicarbonate_min >= 20.0
            then 0
        end as bicarbonate_score
    from (
        select adm.hadm_id, adm.icustay_id, wbc_min, wbc_max, potassium_min, potassium_max, sodium_min, sodium_max, bicarbonate_min, bicarbonate_max
        FROM ADMISSIONS_FULL_METAVISION adm inner join labs_first_day labs on adm.icustay_id = labs.icustay_id        
    ) gg
);



-- gcs score
-- 39871
drop table if exists sapsii_gcs_score;
create table sapsii_gcs_score (
    select t.hadm_id, t.icustay_id, case       
        when mingcs is null then null
        when mingcs <  3 then null -- erroneous value/on trach
        when mingcs <  6 then 26
        when mingcs <  9 then 13
        when mingcs < 11 then 7
        when mingcs < 14 then 5
        when mingcs >= 14
         and mingcs <= 15
          then 0
        end as gcs_score
    from (  
        select DISTINCT adm.hadm_id, adm.icustay_id, mingcs
        FROM ADMISSIONS_FULL_METAVISION adm inner join gcs_first_day gcs on adm.hadm_id = gcs.hadm_id        
    ) t
);

-- comorbidity_score
drop table if exists sapsii_comorbidity_score;
create table sapsii_comorbidity_score (
    select hadm_id, case
            when aids = 1 then 17
            when hem  = 1 then 10
            when mets = 1 then 9
            else 0
        end as comorbidity_score
    from 
    (
    select hadm_id
    -- these are slightly different than elixhauser comorbidities, but based on them
    -- they include some non-comorbid ICD-9 codes (e.g. 20302, relapse of multiple myeloma)
    , max(CASE
        when SUBSTR(icd9_code,1,3) BETWEEN '042' AND '044' THEN 1
            end) as aids      /* HIV and AIDS */
    , max(CASE
        when icd9_code between '20000' and '20238' then 1 -- lymphoma
        when icd9_code between '20240' and '20248' then 1 -- leukemia
        when icd9_code between '20250' and '20302' then 1 -- lymphoma
        when icd9_code between '20310' and '20312' then 1 -- leukemia
        when icd9_code between '20302' and '20382' then 1 -- lymphoma
        when icd9_code between '20400' and '20522' then 1 -- chronic leukemia
        when icd9_code between '20580' and '20702' then 1 -- other myeloid leukemia
        when icd9_code between '20720' and '20892' then 1 -- other myeloid leukemia
        when SUBSTR(icd9_code,1,4) = '2386' then 1 -- lymphoma
        when SUBSTR(icd9_code,1,4) = '2733' then 1 -- lymphoma
            end) as hem
    , max(CASE
        when SUBSTR(icd9_code,1,4) BETWEEN '1960' AND '1991' THEN 1
        when icd9_code between '20970' and '20975' then 1
        when icd9_code = '20979' then 1
        when icd9_code = '78951' then 1
            end) as mets      /* Metastatic cancer */
    from diagnoses_icd
    group by hadm_id
    ) comorb
);


-- admissiontype_score
-- ungrouped size 38919.. admissiontype
-- grouped size 22046
drop table if exists surgflag;
create table surgflag 
(
  select adm.hadm_id
    , case when lower(curr_service) like '%surg%' then 1 else 0 end as surgical
    , ROW_NUMBER() over
    (
      PARTITION BY adm.HADM_ID
      ORDER BY TRANSFERTIME
    ) as serviceOrder
  FROM ADMISSIONS_FULL_METAVISION adm
  left join services se
    on adm.hadm_id = se.hadm_id
);

drop table if exists admissiontype_score_ungrouped;
create table admissiontype_score_ungrouped (
    select t.hadm_id, case
        when admissiontype = 'ScheduledSurgical' then 0
        when admissiontype = 'Medical' then 6
        when admissiontype = 'UnscheduledSurgical' then 8
        else null
      end as admissiontype_score
    from (
    select adm.HADM_ID, case 
          when adm.ADMISSION_TYPE = 'ELECTIVE' and sf.surgical = 1
            then 'ScheduledSurgical'
          when adm.ADMISSION_TYPE != 'ELECTIVE' and sf.surgical = 1
            then 'UnscheduledSurgical'
          else 'Medical'
        end as admissiontype
        from ADMISSIONS_FULL_METAVISION adm left join surgflag sf on adm.hadm_id=sf.hadm_id
    ) t
);


drop table if exists sapsii_admission_type_score;
create table sapsii_admission_type_score (
    select hadm_id, max(admissiontype_score) as admissiontype_score
    from admissiontype_score_ungrouped
    group by hadm_id
);


-- urineoutput
-- 22606
drop table if exists sapsii_uo_score;
create table sapsii_uo_score (
    select t.hadm_id, t.icustay_id, case
      when urineoutput is null then null
      when urineoutput <   500.0 then 11
      when urineoutput <  1000.0 then 4
      when urineoutput >= 1000.0 then 0
    end as uo_score
    from (  
        select DISTINCT adm.hadm_id, uo.icustay_id, urineoutput
        FROM ADMISSIONS_FULL_METAVISION adm inner join urine_output_first_day uo on adm.hadm_id = uo.hadm_id        
    ) t
);

-- merge all
drop table if exists sapsii_score_final;
create table sapsii_score_final (
    select DISTINCT adm.hadm_id, adm.icustay_id
    ,uo_score, wbc_score, bicarbonate_score, sodium_score, potassium_score
    ,pao2fio2_score, hr_score, sysbp_score, temp_score,gcs_score
    ,comorbidity_score, admissiontype_score, age_score 
    FROM ADMISSIONS_FULL_METAVISION adm left join sapsii_uo_score uo on adm.hadm_id = uo.hadm_id
    left join sapsii_lab_score labs on adm.hadm_id = labs.hadm_id
    left join sapsii_paf_score paf on adm.hadm_id = paf.hadm_id
    left join sapsii_vitals_score vit on adm.hadm_id = vit.hadm_id
    left join sapsii_gcs_score gcs on adm.hadm_id = gcs.hadm_id
    left join sapsii_comorbidity_score cs on adm.hadm_id = cs.hadm_id
    left join sapsii_admission_type_score adz on adm.hadm_id = adz.hadm_id
    left join sapsii_age_score age on adm.hadm_id = age.hadm_id
);

drop table if exists sapsii_logistic_regression_label;
create table sapsii_logistic_regression_label (
	select hadm_id, case when DEATHTIME is Null then 0 else 1 end as label from ADMISSIONS_FULL_METAVISION adm
);

select count(DISTINCT saps.icustay_id) FROM sapsii_score_final saps; --23620

-- calculate MAX GROUPED
-- rows 22046
drop table if exists sapsii_score_maxed
create table sapsii_score_maxed (
    select hadm_id, max(uo_score) as uo_score, max(wbc_score) as wbc_score, max(bicarbonate_score) as bicarbonate_score
        ,max(sodium_score) as sodium_score, max(potassium_score) as potassium_score
        ,max(pao2fio2_score) as pao2fio2_score, max(hr_score) as hr_score, max(sysbp_score) as sysbp_score, max(temp_score) as temp_score, max(gcs_score) as gcs_score
        ,max(comorbidity_score) as comorbidity_score, max(admissiontype_score) as admissiontype_score, max(age_score) as age_score
        from sapsii_score_final saps 
        GROUP BY hadm_id 
);

-- averaged
select avg(uo_score) as uo_score, avg(wbc_score) as wbc_score, avg(bicarbonate_score) as bicarbonate_score
    ,avg(sodium_score) as sodium_score, avg(potassium_score) as potassium_score
    ,avg(pao2fio2_score) as pao2fio2_score, avg(hr_score) as hr_score, avg(sysbp_score) as sysbp_score, avg(temp_score) as temp_score, avg(gcs_score) as gcs_score
    ,avg(comorbidity_score) as comorbidity_score, avg(admissiontype_score) as admissiontype_score, avg(age_score) as age_score
    from sapsii_score_maxed;
    

--- Calculate average
-- Merge all
drop table if exists sapsii_raw;
create table sapsii_raw (
    select DISTINCT adm.hadm_id, adm.icustay_id
    ,urineoutput as uo_score, wbc_min, wbc_max, bicarbonate_score, sodium_score, potassium_score
    ,pao2fio2_score, hr_score, sysbp_score, temp_score,gcs_score
    ,comorbidity_score, admissiontype_score, age_score 
    FROM ADMISSIONS_FULL_METAVISION adm left join urine_output_first_day uo on adm.hadm_id = uo.hadm_id
    left join sapsii_lab_score labs on adm.hadm_id = labs.hadm_id
    left join sapsii_paf_score paf on adm.hadm_id = paf.hadm_id
    left join sapsii_vitals_score vit on adm.hadm_id = vit.hadm_id
    left join sapsii_gcs_score gcs on adm.hadm_id = gcs.hadm_id
    left join sapsii_comorbidity_score cs on adm.hadm_id = cs.hadm_id
    left join sapsii_admission_type_score adz on adm.hadm_id = adz.hadm_id
    left join sapsii_age_score age on adm.hadm_id = age.hadm_id
);



