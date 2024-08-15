'''
Config for whole KIDS Engine
'''

from yaml import safe_load
from torch.cuda import is_available
from torch import device

config = safe_load(open("./config.yaml","r",encoding="utf-8"))
device = device('cuda' if is_available() else 'cpu')

########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
RAW_DIR = "./Dataset/CADETS-E3-JSON/"

# The directory to save all artifacts
ARTIFACT_DIR = "./artifact/"

# The directory to save the vectorized graphs
#* GRAPHS_DIR = ARTIFACT_DIR + "graphs/"

# The directory to save the models
#* MODELS_DIR = ARTIFACT_DIR + "models/"
MODEL_NAME = "cadets5_models"
#* MODELS_PATH = MODELS_DIR + MODEL_NAME + ".pt"
# The directory to save the results after testing
#* TEST_RE = ARTIFACT_DIR + "test_re/"

# The directory to save all visualized results
#* VIS_RE = ARTIFACT_DIR + "vis_re/"

# The directory to save logs
LOG_DIR = "./Log/"

########################################################
#
#               Database settings
#
########################################################

# Database name
DATABASE = 'kids_db'

# Only config this setting when you have the problem mentioned
# in the Troubleshooting section in settings/environment-settings.md.
# Otherwise, set it as None
HOST = '/var/run/postgresql/'
# host = None

# Database user
USER = 'postgres'

# The password to the database user
PASSWORD = 'postgres'

# The port number for Postgres
PORT = '5432'

FILE_LIST = [
#    'ta1-cadets-e3-official.json',
#    'ta1-cadets-e3-official.json.1',
#    'ta1-cadets-e3-official.json.2',
    'ta1-cadets-e3-official-1.json',
    'ta1-cadets-e3-official-1.json.1',
    'ta1-cadets-e3-official-1.json.2',
    'ta1-cadets-e3-official-1.json.3',
    'ta1-cadets-e3-official-1.json.4',
#    'ta1-cadets-e3-official-2.json',
#    'ta1-cadets-e3-official-2.json.1'
]

CREATE_PLUGIN = '''
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
'''

CREATE_EVENT_TABLE = f'''
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

alter table event_table owner to {USER};
SELECT create_hypertable('event_table', 'timestamp_data',chunk_time_interval => 86400000000);
ALTER TABLE event_table SET (timescaledb.compress, timescaledb.compress_segmentby = 'src_index_id');

grant delete, insert, references, select, trigger, truncate, update on event_table to {USER};
'''

CREATE_FILE_NODE_TABLE = f'''
create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);


alter table file_node_table owner to {USER};
'''

CREATE_NETFLOW_NODE_TABLE = f'''
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

alter table netflow_node_table owner to {USER};
'''

CREATE_SUBJECT_NODE_TABLE = f'''
create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);

alter table subject_node_table owner to {USER};
'''

CREATE_NODE2ID = f'''
create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);

alter table node2id owner to {USER};

create unique index node2id_hash_id_uindex on node2id (hash_id);
'''

CREATE_ABERRATION_STATICS_TABLE = '''
create table aberration_statics_table
(
    begin_timestamp    timestamp,
    end_timestamp      timestamp,
    begin_time         bigint,
    end_time           bigint,
    loss_avg           double precision,
    count              bigint,
    percentage         double precision,
    node_num           bigint,
    edge_num           bigint
);
'''

CREATE_SUBJECTS_TABLE = '''
create table dangerous_subjects_table
(
    timestamp      timestamp,
    Time           bigint,
    SubjectType    varchar,
    SubjectName    varchar,
    GraphIndex     varchar
);
create table anomalous_subjects_table
(
    timestamp      timestamp,
    Time           bigint,
    SubjectType    varchar,
    SubjectName    varchar,
    GraphIndex     varchar
);
'''

CREATE_ACTIONS_TABLE = '''
create table dangerous_actions_table
(
    timestamp      timestamp,
    Time           bigint,
    SubjectType    varchar,
    SubjectName    varchar,
    Action         varchar,
    OubjectType    varchar,
    OubjectName    varchar,
    GraphIndex     varchar
);
create table anomalous_actions_table
(
    timestamp      timestamp,
    Time           bigint,
    SubjectType    varchar,
    SubjectName    varchar,
    Action         varchar,
    OubjectType    varchar,
    OubjectName    varchar,
    GraphIndex     varchar
);
'''

CREATE_OBJECTS_TABLE = '''
create table dangerous_objects_table
(
    timestamp      timestamp,
    Time           bigint,
    OubjectType    varchar,
    OubjectName    varchar,
    GraphIndex     varchar
);
create table anomalous_objects_table
(
    timestamp      timestamp,
    Time           bigint,
    OubjectType    varchar,
    OubjectName    varchar,
    GraphIndex     varchar
);
'''

DROP_TABLES = '''
DROP TABLE event_table;
DROP TABLE file_node_table;
DROP TABLE netflow_node_table;
DROP TABLE subject_node_table;
DROP TABLE node2id;
DROP TABLE aberration_statics_table;
DROP TABLE dangerous_subjects_table;
DROP TABLE anomalous_subjects_table;
DROP TABLE dangerous_actions_table;
DROP TABLE anomalous_actions_table;
DROP TABLE dangerous_objects_table;
DROP TABLE anomalous_objects_table;
'''

########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
EDGE_REVERSED = [
    "EVENT_ACCEPT",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG"
]

# The following edges are the types only considered to construct the
# temporal graph for experiments.
EDGE_TYPE={
    "cadets-e3":[
        "EVENT_WRITE",
        "EVENT_READ",
        "EVENT_CLOSE",
        "EVENT_OPEN",
        "EVENT_EXECUTE",
        "EVENT_SENDTO",
        "EVENT_RECVFROM", 
    ],
    "low":[    
        'EVENT_CLOSE',
        'EVENT_OPEN',
        'EVENT_READ',
        'EVENT_WRITE',
        'EVENT_EXECUTE',
        'EVENT_RECVFROM',
        'EVENT_RECVMSG',
        'EVENT_SENDMSG',
        'EVENT_SENDTO'
    ],
    "medium":[
        'EVENT_CLOSE',
        'EVENT_CONNECT',
        'EVENT_EXECUTE',
        'EVENT_OPEN',
        'EVENT_READ',
        'EVENT_RECVFROM',
        'EVENT_RECVMSG',
        'EVENT_SENDMSG',
        'EVENT_SENDTO',
        'EVENT_WRITE',
    ],
    "high":[
        'EVENT_CLOSE',
        'EVENT_CONNECT',
        'EVENT_EXECUTE',
        'EVENT_OPEN',
        'EVENT_READ',
        'EVENT_RECVFROM',
        'EVENT_RECVMSG',
        'EVENT_SENDMSG',
        'EVENT_SENDTO',
        'EVENT_WRITE',
        'EVENT_ACCEPT',
        'EVENT_CLONE',
        'EVENT_CREATE_OBJECT',
    ]
}

# The map between edge type and edge ID
REL2ID = {
    "cadets-e3":{ 
        1: 'EVENT_WRITE',
        'EVENT_WRITE': 1,
        2: 'EVENT_READ',
        'EVENT_READ': 2,
        3: 'EVENT_CLOSE',
        'EVENT_CLOSE': 3,
        4: 'EVENT_OPEN',
        'EVENT_OPEN': 4,
        5: 'EVENT_EXECUTE',
        'EVENT_EXECUTE': 5,
        6: 'EVENT_SENDTO',
        'EVENT_SENDTO': 6,
        7: 'EVENT_RECVFROM',
        'EVENT_RECVFROM': 7
    },
    "low":{
        'EVENT_CLOSE':1,
        1:'EVENT_CLOSE',
        'EVENT_OPEN':2,
        2:'EVENT_OPEN',
        'EVENT_READ':3,
        3:'EVENT_READ',
        'EVENT_WRITE':4,
        4:'EVENT_WRITE',
        'EVENT_EXECUTE':5,
        5:'EVENT_EXECUTE',
        'EVENT_RECVFROM':6,
        6:'EVENT_RECVFROM',
        'EVENT_RECVMSG':7,
        7:'EVENT_RECVMSG',
        'EVENT_SENDMSG':8,
        8:'EVENT_SENDMSG',
        'EVENT_SENDTO':9,
        9:'EVENT_SENDTO'
    },
    "medium":{
        'EVENT_CLOSE':1,
        1:'EVENT_CLOSE',
        'EVENT_CONNECT':2,
        2:'EVENT_CONNECT',
        'EVENT_EXECUTE':3,
        3:'EVENT_EXECUTE',
        'EVENT_OPEN':4,
        4:'EVENT_OPEN',
        'EVENT_READ':5,
        5:'EVENT_READ',
        'EVENT_RECVFROM':6,
        6:'EVENT_RECVFROM',
        'EVENT_RECVMSG':7,
        7:'EVENT_RECVMSG',
        'EVENT_SENDMSG':8,
        8:'EVENT_SENDMSG',
        'EVENT_SENDTO':9,
        9:'EVENT_SENDTO',
        'EVENT_WRITE':10,
        10:'EVENT_WRITE',
    },
    "high":{
        'EVENT_CLOSE':1,
        1:'EVENT_CLOSE',
        'EVENT_CONNECT':2,
        2:'EVENT_CONNECT',
        'EVENT_EXECUTE':3,
        3:'EVENT_EXECUTE',
        'EVENT_OPEN':4,
        4:'EVENT_OPEN',
        'EVENT_READ':5,
        5:'EVENT_READ',
        'EVENT_RECVFROM':6,
        6:'EVENT_RECVFROM',
        'EVENT_RECVMSG':7,
        7:'EVENT_RECVMSG',
        'EVENT_SENDMSG':8,
        8:'EVENT_SENDMSG',
        'EVENT_SENDTO':9,
        9:'EVENT_SENDTO',
        'EVENT_WRITE':10,
        10:'EVENT_WRITE',
        'EVENT_ACCEPT':11,
        11:'EVENT_ACCEPT',
        'EVENT_CLONE':12,
        12:'EVENT_CLONE',
        'EVENT_CREATE_OBJECT':13,
        13:'EVENT_CREATE_OBJECT'
    }
}

########################################################
#
#                   Model dimensionality
#
########################################################

# Node Embedding Dimension
NODE_EMBEDDING_DIM = 16

# Node State Dimension
NODE_STATE_DIM = 100

# Neighborhood Sampling Size
NEIGHBOR_SIZE = 20

# Edge Embedding Dimension
EDGE_DIM = 100

# The time encoding Dimension
TIME_DIM = 100

########################################################
#
#                   Train&Test
#
########################################################

# Batch size for training and testing
BATCH = 1024

# Parameters for optimizer
LR = 0.00005
EPS = 1e-08
WEIGHT_DECAY = 0.01

EPOCH_NUM = 50

# The size of time window, 60000000000 represent 1 min in nanoseconds.
# The default setting is 15 minutes.
TIME_WINDOW_SIZE = 60000000000 * 15

########################################################
#
#                   Threshold
#
########################################################

BETA_DAY = 100

#########################################################
#
#
#
#########################################################

KEYWORDS = {
    "cadets-e3":[        
        'netflow',
        '/home/george/Drafts',
        'usr',
        'proc',
        'var',
        'cadet',
        '/var/log/debug.log',
        '/var/log/cron',
        '/home/charles/Drafts',
        '/etc/ssl/cert.pem',
        '/tmp/.31.3022e',
    ],
    "low":[
        'netflow',
        '/home/george/Drafts',
        'usr',
        'proc',
        'var',
        'cadet',
        '/var/log/debug.log',
        '/var/log/cron',
        '/home/charles/Drafts',
        '/etc/ssl/cert.pem',
        '/tmp/.31.3022e',
    ],
    "medium":[
        'netflow',
        '/home/george/Drafts',
        'usr',
        'proc',
        'var',
        'cadet',
        '/var/log/debug.log',
        '/var/log/cron',
        '/home/charles/Drafts',
        '/etc/ssl/cert.pem',
        'sshdlog',
        '/bin/',
        'SOFTWAREPROTECTIONPLATFORM',
        'Windows/Logs/',
        'Windows/system32/',
        'Windows/System32/',
        '/Temp/',
        '/Temp/',
        'Users',
        'USERS',
        'Program Files',
        'WINDOWS',
        'Windows',
        '/home/admin/',
        '/home/user/',
        'proc',
        '/tmp/',
        '/data/system/',
        '/data/data/com.android',
        '/proc/',
        'nz9885vc.default',        
        '/storage/emulated/',
        '/sys/devices/',
        'org.mozilla.fennec_vagrant',
        'mark.via.gp',
        '/data/system_ce/',
        '/Camera',
        'kohimovie.info.kohimovies',
        '.dziauz.tinyflashlight',
        'com.',
        'android.process.media',
    ],
    "high":[
        'sshd',
        'sshdlog',
        'shm',
        'python',
        '189.141.204.211',
        '208.203.20.42',
        'netflow',        
        'salt-minion.log',
        'usr',
        'firefox',
        '/data/replay_logdb',
        '/stat',
        '/boot',
        'qt-opensource-linux-x64',
        '/eraseme',
        '675',
        'null',
        '/dev/pts',
        '/.cache/mozilla/',
        'tmp',
        'thunderbird',
        '/bin/',
        '/sbin/sysctl',
        '/data/replay_logdb/',
        '/home/admin/eraseme',    
        '/stat',
        '.DLL',
        '.dll',
        '.dat', 
        '.DAT', 
        'CACHE',
        'Cache',
        '.docx',
        '.lnk',
        '.LNK',
        '.pptx',
        '.xlsx',
        'CVR',
        'cvr',
        'ZLEAZER',
        'zleazer',
        'SOFTWAREPROTECTIONPLATFORM',
        'documents',
        '.log',
        '.nls',
        '.EVTX',
        '.evtx',
        '.tmp',
        '.TMP',
        'Windows/Logs/',
        'Windows/system32/',
        'Windows/System32/',
        '/Temp/',
        'Users',
        'USERS',
        'Program Files',
        'WINDOWS',
        'Windows',
        '$SII',
        'svchost.exe',
        'gpscript.exe',
        'python.exe',
        'rundll32.exe',
        'consent.exe',
        'python27',
        'Python27',
        '/home/admin/',
        '/home/user/',
        'proc',
        '/tmp/',
        '/var/spool/mqueue/',
        '/var/log/debug.log.0',    
        'glx_alsa_675',
        '/data/system/',
        '/data/data/com.android',
        '/proc/',
        'nz9885vc.default',        
        '/storage/emulated/',
        '/sys/devices/',
        'org.mozilla.fennec_vagrant',
        'mark.via.gp',
        '/data/system_ce/',
        '/Camera',
        'kohimovie.info.kohimovies',
        '.dziauz.tinyflashlight',
        'com.',
        'android.process.media',
        'temp-index',
        '/dev/binder',
        'vanilla'
    ]
}

#########################################################
#
#  attack
#
#########################################################

ATTACK_NODES = {
    "cadets-e3":{
        '/tmp/vUgefal',
        'vUgefal',
        '/var/log/devc',
        '/etc/passwd',
        '81.49.200.166',
        '61.167.39.128',
        '78.205.235.65',
        '139.123.0.113',
        "'nginx'",
    },
    "low":[
        '/tmp/vUgefal',
        'vUgefal',
        '/var/log/devc',
        '/etc/passwd',
        "'nginx'",
        "sysctl",
        "mail",
        "smtpd",
        "service",
        "/bin/sh"
    ],
    "medium":[
        '/tmp/vUgefal',
        'vUgefal',
        '/var/log/devc',
        '/etc/passwd',
        "'nginx'",
        "sysctl",
        "mail",
        "smtpd",
        "service",
        "/bin/sh",
        '/var/log/sshdlog',
        '/usr/sbin/sshd',
        'firefox',
        'sshd',
        'sshdlog',
        'shm',
        '/etc/passwd',
        '/var/log/mail',
        '/var/log',
        './run_webserver.sh',
        'whoami',
        'cat /etc/passwd' 
    ],
    "high":[
        '208.203.20.42',
        '/var/log/sshdlog',
        '/usr/sbin/sshd',
        '81.49.200.166',
        '61.167.39.128',
        '78.205.235.65',
        '139.123.0.113',
        'sshd',
        'sshdlog',
        'shm',
        '189.141.204.211',
        '/home/admin/clean',
        '/dev/glx_alsa_675',
        '/home/admin/profile',
        '/etc/passwd',
        '161.116.88.72',
        '146.153.68.151',
        '/var/log/mail',
        '/tmp/memtrace.so',
        '/tmp',
        '/var/log/xdev',
        '/var/log/wdev',
        'gtcache',
        'firefox',
        '/var/log',
        '145.199.103.57',
        '61.130.69.232',
        '104.228.117.212',
        '141.43.176.203',
        '7.149.198.40',
        '5.214.163.155',
        '149.52.198.23',                    
        '142.20.56.204',
        'lsass.exe',
        '142.20.61.130',
        '132.197.158.98',
        'Credentials',
        'barephone-instr.apk',
        'screencap-instr.apk',
        'de.belu.appstarter',
        './run_webserver.sh',
        'appstarter-instr.apk',
        'screenshot.png',
        'screenshot',
        '/dev/msm_g711tlaw',
        'com.android.providers.contacts',
        'barephone',
        'busybox',
        'screencap',
        '/data/local/tmp',
        'calllog.db',
        'calendar.db',        
        'external.db',
        'internal.db',
        'lastAccess.db',
        'mmssms.db',
        'nginx',
        '128.55.12.167',
        '4.21.51.250',
        'ocMain.py',
        'python',
        '98.23.182.25',
        '108.192.100.31',
        'hostname',
        'whoami',
        'cat /etc/passwd',  
        "'cat'",
        "'scp'",
        "'find'",
        "'bash'",
        "/etc/passwd",
        "/usr/home/user/",
        "128.55.12.167",
        "4.21.51.250",
        "128.55.12.233",
        'shared_files',
        'csb.tracee.27331.27355',
        'netrecon',
        '/data/data/org.mozilla.fennec_firefox_dev/',
        '153.178.46.202',
        '111.82.111.27',
        '166.199.230.185',
        '140.57.183.17',        
        '/data/data/org.mozilla.fennec_firefox_dev/shared_files',
        '/data/data/org.mozilla.fennec_firefox_dev/csb.tracee.27331.27355',
        'glx_alsa_675',
        '77.138.117.150',      
        '128.55.12.33',
        '128.55.12.233',
        '128.55.12.166',
        '49.8.46.240',
        '42.183.7.162',
        '133.39.25.45',                 
    ]
}

REPLACE_DICT = {
    "cadets-e3":{
    '/run/shm/': '/run/shm/*',
    '/home/admin/.cache/mozilla/firefox/': '/home/admin/.cache/mozilla/firefox/*',
    '/home/admin/.mozilla/firefox': '/home/admin/.mozilla/firefox*',
    '/data/replay_logdb/': '/data/replay_logdb/*',
    '/home/admin/.local/share/applications/': '/home/admin/.local/share/applications/*',
    '/usr/share/applications/': '/usr/share/applications/*',
    '/lib/x86_64-linux-gnu/': '/lib/x86_64-linux-gnu/*',
    '/proc/': '/proc/*',
    '/stat': '*/stat',
    '/etc/bash_completion.d/': '/etc/bash_completion.d/*',
    '/usr/bin/python2.7': '/usr/bin/python2.7/*',
    '/usr/lib/python2.7': '/usr/lib/python2.7/*',    
    },
    "low":{
        '/run/shm/': '/run/shm/*',
        '/home/admin/.cache/mozilla/firefox/': '/home/admin/.cache/mozilla/firefox/*',
        '/home/admin/.mozilla/firefox': '/home/admin/.mozilla/firefox*',
        '/data/replay_logdb/': '/data/replay_logdb/*',
        '/home/admin/.local/share/applications/': '/home/admin/.local/share/applications/*',
        '/usr/share/applications/': '/usr/share/applications/*',
        '/lib/x86_64-linux-gnu/': '/lib/x86_64-linux-gnu/*',
        '/proc/': '/proc/*',
        '/stat': '*/stat',
        '/etc/bash_completion.d/': '/etc/bash_completion.d/*',
        '/usr/bin/python2.7': '/usr/bin/python2.7/*',
        '/usr/lib/python2.7': '/usr/lib/python2.7/*',
    },
    "medium":{
        '/run/shm/': '/run/shm/*',
        '/home/admin/.cache/mozilla/firefox/': '/home/admin/.cache/mozilla/firefox/*',
        '/home/admin/.mozilla/firefox': '/home/admin/.mozilla/firefox*',
        '/data/replay_logdb/': '/data/replay_logdb/*',
        '/home/admin/.local/share/applications/': '/home/admin/.local/share/applications/*',
        '/usr/share/applications/': '/usr/share/applications/*',
        '/lib/x86_64-linux-gnu/': '/lib/x86_64-linux-gnu/*',
        '/proc/': '/proc/*',
        '/stat': '*/stat',
        '/etc/bash_completion.d/': '/etc/bash_completion.d/*',
        '/usr/bin/python2.7': '/usr/bin/python2.7/*',
        '/usr/lib/python2.7': '/usr/lib/python2.7/*',
        '/tmp//':'/tmp//*',
        '/home/admin/backup/':'/home/admin/backup/*',
        '/home/admin/./backup/':'/home/admin/./backup/*',
        '/usr/home/admin/./test/':'/usr/home/admin/./test/*',
        '/usr/home/admin/test/':'/usr/home/admin/test/*',
        '/home/admin/out':'/home/admin/out*',    
    },
    "high":{
        '/run/shm/':'/run/shm/*',
        '/home/admin/.cache/mozilla/firefox/pe11scpa.default/cache2/entries/':'/home/admin/.cache/mozilla/firefox/pe11scpa.default/cache2/entries/*',
        '/home/admin/.cache/mozilla/firefox/':'/home/admin/.cache/mozilla/firefox/*',
        '/home/admin/.mozilla/firefox':'/home/admin/.mozilla/firefox*',
        '/data/replay_logdb/':'/data/replay_logdb/*',
        '/home/admin/.local/share/applications/':'/home/admin/.local/share/applications/*',
        '/usr/share/applications/':'/usr/share/applications/*',
        '/lib/x86_64-linux-gnu/':'/lib/x86_64-linux-gnu/*',
        '/proc/':'/proc/*',
        '/stat':'*/stat',
        '/etc/bash_completion.d/':'/etc/bash_completion.d/*',
        '/usr/bin/python2.7':'/usr/bin/python2.7/*',
        '/usr/lib/python2.7':'/usr/lib/python2.7/*',
        '/data/data/org.mozilla.fennec_firefox_dev/cache/':'/data/data/org.mozilla.fennec_firefox_dev/cache/*',
        'UNNAMED':'UNNAMED*',
        '/etc/fonts/':'/etc/fonts/*',       
        '.pyc':'*.pyc',
        '.dll':'*.dll',
        '.DLL':'*.DLL',
        '/usr/ports/':'/usr/ports/*',
        '/usr/home/user/test':'/usr/home/user/test/*',
        '/tmp//':'/tmp//*',
        '/home/admin/backup/':'/home/admin/backup/*',
        '/home/admin/./backup/':'/home/admin/./backup/*',
        '/usr/home/admin/./test/':'/usr/home/admin/./test/*',
        '/usr/home/admin/test/':'/usr/home/admin/test/*',
        '/home/admin/out':'/home/admin/out*',    
        '/data/data/org.mozilla.fennec_firefox_dev/files/':'/data/data/org.mozilla.fennec_firefox_dev/files/*',
        '/system/fonts/':'/system/fonts/*',
        '/data/data/com.android.email/cache/':'/data/data/com.android.email/cache/*',
        '/data/data/com.android.email/files/':'/data/data/com.android.email/files/*',
    }
}

ATTACK_LIST = {
    "cadets-e3":[
    # 4.6    '2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt', # 1 origin
    # 4.6    '2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt', # 1 origin
    # 4.6    '2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt', # 1 origin
    # 4.6    '2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt', # 1 origin
# 2.4966022004832785 ~ 2.6
    # 4.3    '2018-04-03 11:23:30.151962725~2018-04-03 11:38:36.571944378.txt',
    # 4.3    '2018-04-03 13:12:20.341816702~2018-04-03 13:27:31.681797391.txt',
    # 4.3    '2018-04-03 15:48:39.081609492~2018-04-03 16:04:50.021586260.txt',
    # 4.4    '2018-04-04 10:33:58.050097314~2018-04-04 10:49:04.360078648.txt',
    # 4.4    '2018-04-04 11:19:27.870041331~2018-04-04 11:36:12.880014259.txt',
    # 4.4    '2018-04-04 13:26:20.379868292~2018-04-04 13:41:28.539849354.txt',
    # 4.4    '2018-04-04 15:59:38.939661040~2018-04-04 16:14:46.449641533.txt',
    # 4.4    '2018-04-04 22:55:00.039103609~2018-04-04 23:10:10.899084566.txt',
    # 4.5    '2018-04-05 12:53:05.917980125~2018-04-05 13:08:10.457959463.txt',
    # 4.5    '2018-04-05 17:45:00.787588203~2018-04-05 18:00:00.897576901.txt',
    # 4.5    '2018-04-05 18:31:58.647526739~2018-04-05 18:47:07.857504183.txt',
    # 4.5    '2018-04-05 19:34:28.807441594~2018-04-05 19:50:39.457421452.txt',
    # 4.7    '2018-04-07 09:03:01.168052225~2018-04-07 09:19:07.038030612.txt',
    # 4.7    '2018-04-07 09:50:24.827995819~2018-04-07 10:06:03.197964940.txt',
    # 4.7    '2018-04-07 13:23:07.587702371~2018-04-07 13:38:18.347682404.txt',
    # 4.7    '2018-04-07 16:42:51.707437437~2018-04-07 16:57:58.897414026.txt',
    # 4.8    '2018-04-08 09:29:14.306086~2018-04-08 09:44:21.676064.txt',
    # 4.8    '2018-04-08 09:59:49.786050690~2018-04-08 10:15:37.226020700.txt',
    # 4.8    '2018-04-08 16:22:38.675531543~2018-04-08 16:37:53.975513771.txt',
    # 4.8    '2018-04-08 18:56:52.515321710~2018-04-08 19:12:02.515302543.txt',
    # 4.8    '2018-04-08 21:48:19.885095508~2018-04-08 22:04:27.905070991.txt',
    # 4.9    '2018-04-09 13:10:56.683872987~2018-04-09 13:26:02.773834514.txt',
    # 4.9    '2018-04-09 14:58:49.533710740~2018-04-09 15:13:56.753690814.txt',
    # 4.9    '2018-04-09 16:00:00.853635371~2018-04-09 16:15:27.813605709.txt',
    # 4.9    '2018-04-09 16:15:27.813605709~2018-04-09 16:31:35.733588457.txt',
    # 4.10    '2018-04-10 11:10:00.032085407~2018-04-10 11:25:59.612063059.txt',
    # 4.10    '2018-04-10 13:30:01.381898159~2018-04-10 13:45:02.131877082.txt',
    # 4.10    '2018-04-10 14:01:16.261858231~2018-04-10 14:16:23.651836314.txt',
    # 4.10    '2018-04-10 14:47:39.341793994~2018-04-10 15:02:45.981773938.txt',
    # 4.10    '2018-04-10 16:48:40.321632~2018-04-10 17:03:47.701615.txt',
    # 4.11    '2018-04-11 15:06:50.609851240~2018-04-11 16:37:06.967915485.txt',
    # 4.11    '2018-04-11 21:14:52.437540860~2018-04-11 21:29:57.537522725.txt',
    # 4.11    '2018-04-11 21:29:57.537522725~2018-04-11 21:45:08.247498707.txt',
    # 4.11    '2018-04-11 22:00:28.497478385~2018-04-11 22:15:41.307460519.txt',
    # 4.12    '2018-04-12 08:48:39.856615853~2018-04-12 09:04:46.446586633.txt',
    # 4.12    '2018-04-12 13:10:48.496256523~2018-04-12 13:26:52.416234816.txt',
    # 4.12    '2018-04-12 14:44:33.586131454~2018-04-12 15:00:00.046114197.txt',
    # 4.12    '2018-04-12 15:48:06.166044182~2018-04-12 16:04:13.546024402.txt',
#        '2018-04-06 09:31:25.146320510~2018-04-06 09:46:40.966296805.txt', # 2 > 4.5
#        '2018-04-06 10:33:00.136241644~2018-04-06 10:48:11.796216358.txt', # 2 > 4.5
#        '2018-04-06 16:18:33.979401357~2018-04-06 16:34:47.879379108.txt', # 2 > 4.5
#        '2018-04-06 16:50:08.479356930~2018-04-06 17:05:29.239333653.txt', # 2 > 4.5
#        '2018-04-06 01:17:19.056982719~2018-04-06 01:32:29.456961759.txt', # 3 < 2.14
#        '2018-04-06 04:07:44.916766067~2018-04-06 04:22:56.346731444.txt', # 3 < 2.14
#        '2018-04-06 06:11:49.026587588~2018-04-06 06:26:59.116568845.txt', # 3 < 2.14
#        '2018-04-06 23:33:15.408814807~2018-04-06 23:49:23.948792776.txt', # 3 < 2.14
    ],
    "low":[],
    "medium":[],
    "high":[]
}

DEFAULT_SHAPE = "diamond"

BEGIN_TIME = 3392812800000000000
END_TIME = 0
DAY = 86400000000000
HOUR = 3600000000000
QUARTER = 900000000000
MINUTE = 60000000000
TIME_INTERVAL = DAY
LOSS_FACTOR = 1.5
MAX_AVG_LOSS = 10
MIN_AVG_LOSS = 4.7

DETECTION_LEVEL = "cadets-e3"

HELP_MSG = """
Engine [cmd] [args]
cmd:
    -h / --help                                   Get help message for KIDS Engine
    test                                          Test datasets and calculate loss
    analyse                                       Analyse datasets and corresponding loss
    investigate                                   Investigate dangers/anomalies
    run                                           Test, Analyse and Investigate
    api / rpc                                     Work as an API/RPC server
args:
    analyse / investigate / run
        -begin                                    Beginning time of intrusion detection
        -end                                      Ending time of intrusion detection
    api / rpc
        -host                                     Host of KIDS Engine
        -port                                     Port to listen
api:
    ALL responses are in json
    /ping                                         Ping Engine to check if it's survive
    /api/<cmd>/<begin_time>/<end_time>            Execute remote command
    /config/update/<key>/<value>                  Update config parameters
    /config/view                                  View config parameters
"""
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 7777
DEFAULT_DATASET = "CADETS-E3"

ALLOWED_CMD = (
    "run",
    "test",
    "analyse",
    "investigate"
)

FORBIDDEN_KEYS = (
    "EDGE_REVERSED",
    "EDGE_TYPE",
    "REL2ID",
    "FILE_LIST",
    "ATTACK_LIST"
)
