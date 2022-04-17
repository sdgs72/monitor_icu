create TABLE ventilation_first_day (
    subject_id mediumint unsigned NOT NULL,
    hadm_id mediumint unsigned NOT NULL,
    icustay_id mediumint unsigned NOT NULL,
    vent mediumint unsigned NOT NULL
);

insert into ventilation_first_day(
select
  ie.subject_id, ie.hadm_id, ie.icustay_id
  -- if vd.icustay_id is not null, then they have a valid ventilation event
  -- in this case, we say they are ventilated
  -- otherwise, they are not
  , max(case
      when vd.icustay_id is not null then 1
    else 0 end) as vent
FROM icustays ie
left join ventilation_durations vd
  on ie.icustay_id = vd.icustay_id
  and
  (
    -- ventilation duration overlaps with ICU admission -> vented on admission
    (vd.starttime <= ie.intime and vd.endtime >= ie.intime)
    -- ventilation started during the first day
    OR (vd.starttime >= ie.intime and vd.starttime <= DATE_ADD(ie.intime, INTERVAL 1 DAY))
  )
group by ie.subject_id, ie.hadm_id, ie.icustay_id
order by ie.subject_id, ie.hadm_id, ie.icustay_id
);