


# row counts.......
select count(adm.HADM_ID) from mimiciiiv14.ADMISSIONS as adm; #58976
select count(icu.HADM_ID) from mimiciiiv14.ICUSTAYS as icu; #61532
select count(DISTINCT icu.HADM_ID) from mimiciiiv14.ICUSTAYS as icu; #57786


select count(DISTINCT HADM_ID) from ADMISSIONS_FULL_METAVISION where DBVERSION='metavision' #22049
select count(DISTINCT HADM_ID) from ADMISSIONS_FULL_METAVISION afm where DEATHTIME is not null; #2323 -- unique admission
select count(DISTINCT ICUSTAY_ID) from ADMISSIONS_FULL_METAVISION afm where DEATHTIME is not null; #23620 -- unqiue icu stays



# check if there are any two ADM ID with different sources... (there is none!)
select i.HADM_ID, count(DISTINCT DBSOURCE) m, count(*) c from ICUSTAYS i group by i.HADM_ID HAVING (c>1 and m>1);



# combine them too ADMISSIONS TABLE
CREATE TABLE ICU_JOIN_TABLE as
select adm.*, icu.* from mimiciiiv14.admissions as adm 
join mimiciiiv14.ICUSTAYS icu on adm.HADM_ID = icu.HADM_ID
where icu.DBSOURCE='metavision';
 

# create my admissions table
CREATE TABLE ADMISSIONS as
select adm.HADM_ID, adm.ADMITTIME, adm.DISCHTIME, adm.DEATHTIME, adm.DBSOURCE as DBSOURCE from mimiciiiv14.ADMISSIONS as adm 
where adm.DBSOURCE = 'metavision';

#DEATH_TABLE
drop table if exists DEATH_EVENTS;
CREATE TABLE DEATH_EVENTS (
select adm.HADM_ID,
'Death' as EventType,
case when adm.DEATHTIME is not null then 1 else 0 end as ITEMID,
case when adm.DEATHTIME is not null then 1 else 0 end as ITEMID2,
adm.ADMITTIME as EventStartTime,
case when adm.DEATHTIME is not null then timestampdiff(hour,adm.ADMITTIME,adm.DEATHTIME) else timestampdiff(hour,adm.ADMITTIME,adm.DISCHTIME) end as Time_to_Discharge 
from mimiciiiv14.ADMISSIONS_FULL_METAVISION adm
);
-- timestampdiff(hour,adm.ADMITTIME,adm.DISCHTIME) as Time_to_Discharge


# LAB EVENTS < 3 mins
drop table if exists LAB_EVENTS;
CREATE TABLE LAB_EVENTS as
select adm.HADM_ID,
#adm.DISCHTIME,adm.DEATHTIME,
'Lab' as EventType, lab.ITEMID,
case
  when FLAG is not null then concat(lab.ITEMID, lab.FLAG)
  else lab.ITEMID
end as ITEMID2,
lab.CHARTTIME as EventStartTime,
timestampdiff(hour,lab.CHARTTIME, adm.DISCHTIME) as Time_to_Discharge
from mimiciiiv14.ADMISSIONS adm
left join mimiciiiv14.LABEVENTS lab on adm.HADM_ID = lab.HADM_ID;

# MED EVENTS < 3mins
drop table if exists MED_EVENTS;
CREATE TABLE MED_EVENTS as
select adm.HADM_ID,
#adm.DISCHTIME,adm.DEATHTIME,
'Med' as EventType, med.ITEMID, med.ITEMID as ITEMID2,med.STARTTIME as EventStartTime,
timestampdiff(hour,med.STARTTIME,  adm.DISCHTIME) as Time_to_Discharge
from mimiciiiv14.ADMISSIONS adm
left join mimiciiiv14.INPUTEVENTS_MV med on adm.HADM_ID = med.HADM_ID;



# VIT EVENTS
CREATE TABLE mimiciiiv14.MY_VIT_EVENTS (
	HADM_ID mediumint unsigned NOT NULL,
	EventType varchar(5) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT '' NOT NULL,
	ITEMID varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
	ITEMID2 varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
	EventStartTime datetime NOT NULL,
	Time_to_Discharge bigint NULL
)

INSERT INTO MY_VIT_EVENTS
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
join mimiciiiv14.CHARTEVENTS vit on adm.HADM_ID = vit.HADM_ID;



create table CS3750_Group2.KY_ADM_LENGTH as
set @row_number = 0;
select (@row_number:=@row_number + 1) as row,
t.* from
(select HADM_ID, -1 * min(Time_to_Discharge) adm_length
 from DEATH_EVENTS
 group by HADM_ID) t
;


#Vocabulary calculation
select count(distinct ITEMID) from MED_EVENTS; #277
select count(distinct ITEMID) from LAB_EVENTS; #666
select count(distinct ITEMID2) from LAB_EVENTS; #931
select count(distinct ITEMID, ITEMID2) from LAB_EVENTS; #931
select count(distinct ITEMID, ITEMID2) from MY_VIT_EVENTS; #2798



# All joinedevents
drop table if exists JOINED_EVENTS;
CREATE TABLE JOINED_EVENTS (
	HADM_ID mediumint unsigned NOT NULL,
	EventType varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
	ITEMID varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
	ITEMID2 varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
	EventStartTime datetime NOT NULL,
	Time_to_Discharge bigint NOT NULL
)

INSERT INTO JOINED_EVENTS select * from DEATH_EVENTS;
INSERT INTO JOINED_EVENTS select * from LAB_EVENTS;
INSERT INTO JOINED_EVENTS select * from MED_EVENTS;
-- INSERT INTO JOINED_EVENTS select * from MY_VIT_EVENTS;


SELECT * from ky_mimic_events_v4 INTO OUTFILE 'MIMIC_FULL_BATCH.csv' FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' ESCAPED BY '' LINES TERMINATED BY '\n';
