'''
Utils for KIDS Engine
'''

import pytz
from time import mktime
from datetime import datetime
import time
import psycopg2
from psycopg2 import extras as ex
import os
import copy
import torch
from torch.nn import Linear
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import TemporalData
from torch_geometric.nn import TGNMemory, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage, MeanAggregator,
                                           LastAggregator)
from torch_geometric import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import math
import copy
import time
import xxhash
import gc
from time import ctime

from config import config

#################################################
#* General

class Command(object):
    '''
    Command for KIDS Engine
    '''
    def __init__(self) -> None:
        self.help = False
        self.cmd = None
        self.api_args = {
            "host": config["DEFAULT_HOST"],
            "port": config["DEFAULT_PORT"]
        }
        self.init_args = {}
        self.begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def parse(self, args: list[str]):
        '''
        parse arguments
        '''
        self.cmd = args[1].lower()
        if self.cmd in ("-h", "--help"):
            self.help = True
            return
        elif self.cmd in ("run","investigate","analyse","test"):
            for index, arg in enumerate(args):
                try:
                    if arg.lower() in ("-begin","--begin"):
                        self.begin_time = f"{args[index+1]} 00:00:00"
                        if ":" in args[index + 2]:
                            self.begin_time = f"{args[index+1]} {args[index+2]}"
                    if arg.lower() in ("-end","--end"):
                        self.end_time = f"{args[index+1]} 00:00:00"
                        if ":" in args[index + 2]:
                            self.end_time = f"{args[index+1]} {args[index+2]}"
                except IndexError:
                    pass
        elif self.cmd in ("init"):
            pass
        elif self.cmd in ("rpc", "flask", "api"):
            if len(args) == 2:
                return
            flag = True
            for index, arg in enumerate(args):
                try:
                    if arg in ("-host","-h","--h","--host"):
                        self.api_args["host"] = args[index+1].lower()
                        flag = False
                    if arg in ("-port","-p","--p","--port"):
                        self.api_args["port"] = args[index+1]
                        flag = False
                except IndexError:
                    pass
            if flag:
                self.api_args["host"] = args[2].lower()
                self.api_args["port"] = args[3]

def ns_time_to_datetime(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    dt = datetime.fromtimestamp(int(ns) // 1000000000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def time_to_datetime_US(s):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(s), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')

    return s

def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00.000000000
    :return: nano timestamp
    """
    date , ns_sec = date.split('.')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000 + int(ns_sec)
    return timeStamp

def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00.000000000
    :return: nano timestamp
    """
    date , ns_sec = date.split('.')
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000 + int(ns_sec)
    return int(timeStamp)

def datetime_to_timestamp_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp
    return int(timeStamp)

def init_database_connection():
    '''
    init_database_connection
    '''
    if config.get("HOST",None):
        connect = psycopg2.connect(database = config["DATABASE"],
                                   host = config["HOST"],
                                   user = config["USER"],
                                   password = config["PASSWORD"],
                                   port = config["PORT"]
                                  )
    else:
        connect = psycopg2.connect(database = config["DATABASE"],
                                   user = config["USER"],
                                   password = config["PASSWORD"],
                                   port = config["PORT"]
                                  )
    cur = connect.cursor()
    return cur, connect

def gen_nodeid2msg(cur):
    '''
    从 node2id 表中读取所有字段并生成 index_id:{node_type:msg} 的字典
    '''
    sql = "select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}
    for i in rows:
        nodeid2msg[i[0]] = i[-1]
        nodeid2msg[i[-1]] = {i[1]: i[2]}
    return nodeid2msg

def tensor_find(t,x):
    '''
    tensor_find
    '''
    t_np=t.cpu().numpy()
    idx=np.argwhere(t_np==x)
    return idx[0][0]+1

def std(t):
    '''
    std
    '''
    t = np.array(t)
    return np.std(t)

def var(t):
    '''
    var
    '''
    t = np.array(t)
    return np.var(t)

def mean(t):
    '''
    mean
    '''
    t = np.array(t)
    return np.mean(t)

def hashgen(l):
    """
    Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value.
    """
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()

#################################################

#################################################
#* embedder

def get_events(cur,begin_time,end_time):
    '''
    get_events
    '''
    sql = """
    select * from event_table
    where
          timestamp_rec>'%s' and timestamp_rec<'%s'
           ORDER BY timestamp_rec;
    """ % (begin_time, end_time)
    cur.execute(sql)
    events = cur.fetchall()
    return events

def path2higlist(p):
    '''
    path2higlist
    '''
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
    return l

def ip2higlist(p):
    '''
    ip2higlist
    '''
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
    return l

def list2str(l):
    '''
    list2str
    '''
    return ''.join(l)

#################################################

#################################################
#* investigator

def get_attack_list(cur,begin_time,end_time):
    '''
    get_attack_list
    '''
    attack_list = config["ATTACK_LIST"][config["DETECTION_LEVEL"]]
    if not attack_list:
    #*attack_list = os.listdir(f"{ARTIFACT_DIR}/graph_list")

    #* for file in os.listdir(f"{ARTIFACT_DIR}/graph_list"):
    #*    attack_list.append(f"{ARTIFACT_DIR}/graph_list/{file}")
        attack_list = {}
        sql = """
        select * from aberration_statics_table;
        """
        cur.execute(sql)
        results = cur.fetchall()
        for result in results:
            if result[2] > config["MIN_AVG_LOSS"] and result[2] < config["MAX_AVG_LOSS"] and \
                result[0] > begin_time and result[1] < end_time:
                attack_list[
                    f"{ns_time_to_datetime_US(result[0])}~{ns_time_to_datetime_US(result[1])}.txt\n"
                ] = 1
        attack_list = attack_list.keys()
    return attack_list

def replace_path_name(path_name):
    '''
    replace_path_name
    '''
    for i in config["REPLACE_DICT"][config["DETECTION_LEVEL"]]:
        if i in path_name:
            return config["REPLACE_DICT"][config["DETECTION_LEVEL"]][i]
    return path_name

def save_dangerous_actions(cur,connect,dangerous_action_list):
    '''
    save_dangerous_actions
    '''
    sql = '''insert into dangerous_actions_table
                         values %s
            '''
    ex.execute_values(cur, sql, dangerous_action_list, page_size=10000)
    connect.commit()

def save_dangerous_subjects(cur,connect,dangerous_subjects):
    '''
    save_dangerous_subjects
    '''
    sql = '''insert into dangerous_subjects_table
                         values %s
            '''
    ex.execute_values(cur, sql, dangerous_subjects, page_size=10000)
    connect.commit()

def save_dangerous_objects(cur,connect,dangerous_objects):
    '''
    save_dangerous_objects
    '''
    sql = '''insert into dangerous_objects_table
                         values %s
            '''
    ex.execute_values(cur, sql, dangerous_objects, page_size=10000)
    connect.commit()

def save_anomalous_actions(cur,connect,anomalous_actions):
    '''
    save_anomalous_actions
    '''
    sql = '''insert into anomalous_actions_table
                         values %s
            '''
    ex.execute_values(cur, sql, anomalous_actions, page_size=10000)
    connect.commit()

def save_anomalous_subjects(cur,connect,anomalous_subjects):
    '''
    save_anomalous_subjects
    '''
    sql = '''insert into anomalous_subjects_table
                         values %s
            '''
    ex.execute_values(cur, sql, anomalous_subjects, page_size=10000)
    connect.commit()

def save_anomalous_objects(cur,connect,anomalous_objects):
    '''
    save_anomalous_objects
    '''
    sql = '''insert into anomalous_objects_table
                         values %s
            '''
    ex.execute_values(cur, sql, anomalous_objects, page_size=10000)
    connect.commit()

def save_aberration_statics(
    cur,
    connect,
    aberration_statics
):
    '''
    save_aberration_statics
    '''
    sql = '''insert into aberration_statics_table
                         values %s
            '''
    ex.execute_values(cur, sql, aberration_statics, page_size=10000)
    connect.commit()

#################################################
