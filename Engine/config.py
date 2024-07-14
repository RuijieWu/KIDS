'''
Config for whole KIDS Engine
'''

########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
RAW_DIR = "/home/postgres/KIDS/Dataset/CADETS-E3-JSON/"

# The directory to save all artifacts
ARTIFACT_DIR = "/home/postgres/KIDS/artifact/"

# The directory to save the vectorized graphs
GRAPHS_DIR = ARTIFACT_DIR + "graphs/"

# The directory to save the models
MODELS_DIR = ARTIFACT_DIR + "models/"
MODEL_NAME = "cadets3_models"
MODELS_PATH = MODELS_DIR + MODEL_NAME + ".pt"
# The directory to save the results after testing
TEST_RE = ARTIFACT_DIR + "test_re/"

# The directory to save all visualized results
VIS_RE = ARTIFACT_DIR + "vis_re/"

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
'''

CREATE_ACTIONS_TABLE = '''
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
'''

CREATE_OBJECTS_TABLE = '''
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
        "/bin/sh"
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
        'cat /etc/passwd',  
        
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
        '2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt',
        '2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt',
        '2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt',
        '2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt'
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
MAX_AVG_LOSS = 100
MIN_AVG_LOSS = 4.5

DETECTION_LEVEL = "cadets-e3"

HELP_MSG = ""
