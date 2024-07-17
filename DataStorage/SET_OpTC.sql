# execute the psql with postgres user
sudo -u postgres psql

# create the database
create database optc_db;

# switch to the created database
\connect optc_db;

# create the event table and grant the privileges to postgres
create table event_table
(
    src_id     varchar,
    src_type   varchar,
    edge_type  varchar,
    dst_id     varchar,
    dst_type   varchar,
    hostname   varchar,
    timestamp  bigint,
    data_label varchar
);
alter table event_table owner to postgres;

# create the node2id table
create table nodeid2msg
(
    node_id varchar,
    msg     varchar
);
alter table nodeid2msg owner to postgres;
