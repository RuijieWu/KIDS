# DataBase Setting

## Engine 数据库配置（cadets-e3）

这些数据库配置也可以由 KIDS Engine 通过 init 参数自行创建

### 进入postgresql

```bash
sudo -u postgres psql
```

### 创建数据库

```sql
drop database kids_db;
create database kids_db;
```

### 连接数据库

```sql
\c kids_db;
```

### 创建时序数据库插件

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
```

### 创建event_table

```sql
create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    timestamp_data timestamp,
    _id           serial
);

create unique index event_table__id_uindex on event_table(_id,timestamp_data);

alter table event_table owner to postgres;
SELECT create_hypertable('event_table', 'timestamp_data',chunk_time_interval => 86400000000);
ALTER TABLE event_table SET (timescaledb.compress, timescaledb.compress_segmentby = 'src_index_id');

grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;
```

### 创建file_table

```sql
create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);


alter table file_node_table owner to postgres;
```

### 创建netflow_node_table

```sql
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
```

### 创建subject_node_table

```sql
create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);

alter table subject_node_table owner to postgres;
```

### 创建node2id

```sql
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
```

### 创建aberration_statics_table

```sql
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
```

### 创建subjects_table

```sql
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
```

### 创建actions_table

```sql
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
```

### 创建objects_table

```sql
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
```

## 其它数据集数据库

参考当前目录下的其它 .sql 批处理文件
