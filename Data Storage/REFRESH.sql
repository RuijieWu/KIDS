drop table aberration_statics_table;
create table aberration_statics_table
(
    begin_time         bigint,
    end_time           bigint,
    loss_avg           double precision,
    count              bigint,
    percentage         double precision,
    node_num           bigint,
    edge_num           bigint
);

drop table dangerous_subjects_table;
drop table anomalous_subjects_table;
create table dangerous_subjects_table
(
    Time      bigint,
    SubjectType    varchar,
    SubjectName    varchar
);
create table anomalous_subjects_table
(
    Time      bigint,
    SubjectType    varchar,
    SubjectName    varchar
);

drop table dangerous_actions_table;
drop table anomalous_actions_table;
create table dangerous_actions_table
(
    Time           bigint,
    SubjectType    varchar,
    SubjectName    varchar,
    Action         varchar,
    OubjectType    varchar,
    OubjectName    varchar
);
create table anomalous_actions_table
(
    Time           bigint,
    SubjectType    varchar,
    SubjectName    varchar,
    Action         varchar,
    OubjectType    varchar,
    OubjectName    varchar
);

drop table dangerous_objects_table;
drop table anomalous_objects_table;

create table dangerous_objects_table
(
    Time           bigint,
    OubjectType    varchar,
    OubjectName    varchar
);
create table anomalous_objects_table
(
    Time           bigint,
    OubjectType    varchar,
    OubjectName    varchar
);