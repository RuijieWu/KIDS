# execute the psql with postgres user
sudo -u postgres psql

# create the database
create database tc_theia_dataset_db;

# switch to the created database
\connect tc_theia_dataset_db;

# create the event table and grant the privileges to postgres
create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
alter table event_table owner to postgres;
create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
alter table file_node_table owner to postgres;

# create the netflow table
create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
alter table netflow_node_table owner to postgres;

# create the subject table
create table subject_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    "cmdLine" varchar not null,
    tgid      varchar not null,
    path      varchar not null,
    constraint subject_node_table_pk
        primary key (node_uuid, hash_id)
);
alter table subject_node_table owner to postgres;

# create the node2id table
create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
alter table node2id owner to postgres;
create unique index node2id_hash_id_uindex on node2id (hash_id);
