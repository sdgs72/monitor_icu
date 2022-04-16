#create master MIMIC event table
CREATE TABLE CS3750_Group2.KY_MIMIC_EVENTS_V4 as
select
HADM_ID,
EventType,ITEMID,ITEMID2,EventStartTime, Time_to_Discharge
from
(
select adm.HADM_ID,
#adm.DISCHTIME,adm.DEATHTIME,
'Death' as EventType,
case when adm.DEATHTIME is not null then 1 else 0 end as ITEMID,
case when adm.DEATHTIME is not null then 1 else 0 end as ITEMID2,
adm.ADMITTIME as EventStartTime,
timestampdiff(hour,adm.DISCHTIME,adm.DISCHTIME) as Time_to_Discharge
from mimiciiiv14.ADMISSIONS adm
where adm.DBSOURCE = 'metavision'
#and adm.HADM_ID = 105017

union all

select
sep.HADM_ID,
'Sepsis' as EventType,
1 as ITEMID,
1 as ITEMID2,
sep.STARTTIME as EventStartTime,
timestampdiff(hour,adm.DISCHTIME,sep.STARTTIME) as Time_to_Discharge
from mimiciiiv14.SEPSIS_TABLE as sep
join mimiciiiv14.ADMISSIONS adm on adm.HADM_ID = sep.HADM_ID
where adm.DBSOURCE = 'metavision'
#and adm.HADM_ID = 105017

union all

select
aki.HADM_ID,
'AKI' as EventType,
AKI_LEVEL as ITEMID,
AKI_LEVEL as ITEMID2,
aki.CHARTTIME as EventStartTime,
timestampdiff(hour,adm.DISCHTIME, aki.CHARTTIME) as Time_to_Discharge
from (
select
HADM_ID,AKI_LEVEL,min(CHARTTIME) CHARTTIME
from mimiciiiv14.AKI_TABLE
group by HADM_ID,AKI_LEVEL) aki
join mimiciiiv14.ADMISSIONS adm on adm.HADM_ID = aki.HADM_ID
where adm.DBSOURCE = 'metavision'
#and adm.HADM_ID = 105017

union all

select adm.HADM_ID,
#adm.DISCHTIME,adm.DEATHTIME,
'Lab' as EventType, lab.ITEMID,
case
  when FLAG is not null then concat(lab.ITEMID, lab.FLAG)
  else lab.ITEMID
end as ITEMID2,
lab.CHARTTIME as EventStartTime,
timestampdiff(hour,adm.DISCHTIME,lab.CHARTTIME) as Time_to_Discharge
from mimiciiiv14.ADMISSIONS adm
left join mimiciiiv14.LABEVENTS lab on adm.HADM_ID = lab.HADM_ID
where adm.DBSOURCE = 'metavision' and lab.ITEMID is not null
#and adm.HADM_ID = 105017

union all

select adm.HADM_ID,
#adm.DISCHTIME,adm.DEATHTIME,
'Med' as EventType, med.ITEMID, med.ITEMID as ITEMID2,med.STARTTIME as EventStartTime,
timestampdiff(hour, adm.DISCHTIME, med.STARTTIME) as Time_to_Discharge
from mimiciiiv14.ADMISSIONS adm
left join mimiciiiv14.INPUTEVENTS_MV med on adm.HADM_ID = med.HADM_ID
where adm.DBSOURCE = 'metavision' and med.ITEMID is not null
#and adm.HADM_ID = 105017

union all

select adm.HADM_ID,
#adm.DISCHTIME,adm.DEATHTIME,
'Vit' as EventType, vit.ITEMID,
case
    when WARNING = 1 then concat(vit.ITEMID,'W')
    else vit.ITEMID
end as ITEMID2,
vit.CHARTTIME as EventStartTime,
timestampdiff(hour, adm.DISCHTIME, vit.CHARTTIME) as Time_to_Discharge
from mimiciiiv14.ADMISSIONS adm
left join mimiciiiv14.CHARTEVENTS vit on adm.HADM_ID = vit.HADM_ID
where adm.DBSOURCE = 'metavision' and vit.ITEMID is not null
#and adm.HADM_ID = 105017
    ) T
order by HADM_ID,Time_to_Discharge
;



#create admission length table
set @row_number = 0;
drop table CS3750_Group2.KY_ADM_LENGTH;
create table CS3750_Group2.KY_ADM_LENGTH as
select (@row_number:=@row_number + 1) as row,
t.* from
(select HADM_ID, -1 * min(Time_to_Discharge) adm_length
 from CS3750_Group2.KY_MIMIC_EVENTS_V4
 group by HADM_ID) t
;


#create reference table
create table CS3750_Group2.KY_LABEL_REF as
select *
from (
select 'Lab' as EventType, l.ITEMID2, d.LABEL
from (select distinct ITEMID, ITEMID2 from CS3750_Group2.KY_MIMIC_EVENTS_V4 where EventType = 'Lab') l
left join mimiciiiv14.D_LABITEMS d on l.ITEMID = d.ITEMID

union all

select 'Med' as EventType,l.ITEMID2, d.LABEL
from (select distinct ITEMID,ITEMID2 from CS3750_Group2.KY_MIMIC_EVENTS_V4 where EventType = 'Med') l
left join mimiciiiv14.D_ITEMS d on l.ITEMID = d.ITEMID
where d.DBSOURCE = 'metavision'

union all

select 'Vit' as EventType,l.ITEMID2, d.LABEL
from (select distinct ITEMID, ITEMID2 from CS3750_Group2.KY_MIMIC_EVENTS_V4 where EventType = 'Vit') l
left join mimiciiiv14.D_ITEMS d on l.ITEMID = d.ITEMID
where d.DBSOURCE = 'metavision'
)
t;