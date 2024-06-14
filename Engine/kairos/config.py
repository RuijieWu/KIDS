"""
Config for whole GNN Engine
"""
########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
RAW_DIR = "/home/postgres/kairos/CADETS-E3"

# The directory to save all artifacts
ARTIFACT_DIR = "./artifact/"

# The directory to save the vectorized graphs
GRAPHS_DIR = ARTIFACT_DIR + "graphs/"

# The directory to save the models
MODELS_DIR = ARTIFACT_DIR + "models/"
MODELS_PATH = MODELS_DIR + "cadets3_models.pt"

# The directory to save the results after testing
TEST_RE = ARTIFACT_DIR + "test_re/"

# The directory to save all visualized results
VIS_RE = ARTIFACT_DIR + "vis_re/"



########################################################
#
#               Database settings
#
########################################################

# Database name
DATABASE = 'tc_cadet_dataset_db'

# Only config this setting when you have the problem mentioned
# in the Troubleshooting section in settings/environment-settings.md.
# Otherwise, set it as None
HOST = '/var/run/postgresql/'
# host = None

# Database user
USER = 'postgres'

# The password to the database user
PASSWORD = 'JeRyWu'

# The port number for Postgres
PORT = '5432'

FILE_LIST = [
    'ta1-cadets-e3-official.json',
    'ta1-cadets-e3-official.json.1',
    'ta1-cadets-e3-official.json.2',
    'ta1-cadets-e3-official-1.json',
    'ta1-cadets-e3-official-1.json.1',
    'ta1-cadets-e3-official-1.json.2',
    'ta1-cadets-e3-official-1.json.3',
    'ta1-cadets-e3-official-1.json.4',
    'ta1-cadets-e3-official-2.json',
    'ta1-cadets-e3-official-2.json.1'
]



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
EDGE_TYPE=[
    "EVENT_WRITE",
    "EVENT_READ",
    "EVENT_CLOSE",
    "EVENT_OPEN",
    "EVENT_EXECUTE",
    "EVENT_SENDTO",
    "EVENT_RECVFROM",
]

# The map between edge type and edge ID
REL2ID = {
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

BETA_DAY6 = 100
BETA_DAY7 = 100

#########################################################
#
#
#
#########################################################

DETECTION_LEVEL = "low"

KEYWORDS = {
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
        
    ],
    "high":[
        
    ]
}

#########################################################
#
#  attack
#
#########################################################
ATTACK_NODES = {
    "low":[
        '/tmp/vUgefal',
        'vUgefal',
        '/var/log/devc',
        '/etc/passwd',
        '81.49.200.166',
        '61.167.39.128',
        '78.205.235.65',
        '139.123.0.113',
        "'nginx'",
    ],
    "medium":[
        
    ],
    "high":[
        
    ]
}

REPLACE_DICT = {
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
}

ATTACK_LIST = [
    ARTIFACT_DIR+'/graph_4_6/2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt',
    ARTIFACT_DIR+'/graph_4_6/2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt',
    ARTIFACT_DIR+'/graph_4_6/2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt',
    ARTIFACT_DIR+'/graph_4_6/2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt',
]
