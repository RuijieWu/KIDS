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

from config import *

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
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    return timeStamp

def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
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
    if HOST is not None:
        connect = psycopg2.connect(database = DATABASE,
                                   host = HOST,
                                   user = USER,
                                   password = PASSWORD,
                                   port = PORT
                                  )
    else:
        connect = psycopg2.connect(database = DATABASE,
                                   user = USER,
                                   password = PASSWORD,
                                   port = PORT
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
    t_np=t.cpu().numpy()
    idx=np.argwhere(t_np==x)
    return idx[0][0]+1

def std(t):
    t = np.array(t)
    return np.std(t)

def var(t):
    t = np.array(t)
    return np.var(t)

def mean(t):
    t = np.array(t)
    return np.mean(t)

def hashgen(l):
    """Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value."""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()

#################################################
#* embedder

def get_events(cur,begin_time,end_time):
    start_timestamp = datetime_to_ns_time_US(begin_time)
    end_timestamp = datetime_to_ns_time_US(end_time)
    sql = """
    select * from event_table
    where
          timestamp_rec>'%s' and timestamp_rec<'%s'
           ORDER BY timestamp_rec;
    """ % (start_timestamp, end_timestamp)
    cur.execute(sql)
    events = cur.fetchall()
    return events

def path2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
    return l

def ip2higlist(p):
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
    return l

def list2str(l):
    return ''.join(l)
#################################################

#################################################
#* investigator

def replace_path_name(path_name):
    for i in REPLACE_DICT[DETECTION_LEVEL]:
        if i in path_name:
            return REPLACE_DICT[DETECTION_LEVEL][i]
    return path_name

def save_dangerous_actions(cur,connect,dangerous_action_list):
    sql = '''insert into  anomalous_actions_table
                         values %s
            '''
    ex.execute_values(cur, sql, dangerous_action_list, page_size=10000)
    connect.commit()

def save_dangerous_subjects(cur,connect,dangerous_subjects):
    sql = '''insert into dangerous_subjects_table
                         values %s
            '''
    ex.execute_values(cur, sql, dangerous_subjects, page_size=10000)
    connect.commit()

def save_dangerous_objects(cur,connect,dangerous_objects):
    sql = '''insert into  dangerous_objects_table
                         values %s
            '''
    ex.execute_values(cur, sql, dangerous_objects, page_size=10000)
    connect.commit()


def save_anomalous_actions(cur,connect,anomalous_actions):
    sql = '''insert into  dangerous_actions_table
                         values %s
            '''
    ex.execute_values(cur, sql, anomalous_actions, page_size=10000)
    connect.commit()

def save_anomalous_subjects(cur,connect,anomalous_subjects):
    sql = '''insert into anomalous_subjects_table
                         values %s
            '''
    ex.execute_values(cur, sql, anomalous_subjects, page_size=10000)
    connect.commit()

def save_anomalous_objects(cur,connect,anomalous_objects):
    sql = '''insert into  anomalous_objects_table
                         values %s
            '''
    ex.execute_values(cur, sql, anomalous_objects, page_size=10000)
    connect.commit()

def save_aberration_statics(
    cur,
    connect,
    path,
    loss_avg,
    count,
    percentage,
    node_num,
    edge_num
):
    datalist = []
    datalist.append([
        path,
        loss_avg,
        count,
        percentage,
        node_num,
        edge_num
    ])
    sql = '''insert into aberration_statics_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()
#################################################
