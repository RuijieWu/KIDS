#!/usr/bin/env python
# coding: utf-8

# In[2]:


# encoding=utf-8
import os.path as osp
import os
import copy
import matplotlib.pyplot as plt
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
from torch_geometric.utils import negative_sampling

from tqdm import tqdm
# from .autonotebook import tqdm as notebook_tqdm

import networkx as nx
import numpy as np
import math
import copy
import re
import time
import json
import pandas as pd
from random import choice
import gc
from graphviz import Digraph
import xxhash

from datetime import datetime, timezone
import time
import pytz
from time import mktime
from datetime import datetime
import time


from rich.progress import Progress
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def hashgen(l):
    """Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value."""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()


def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    date,ns=date.split('.')

    timeArray = time.strptime(date, '%Y-%m-%dT%H:%M:%S')
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    timeStamp += int(ns.split('Z')[0])
    return timeStamp


def datetime_to_timestamp_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    date=date.replace('-04:00','')
    if '.' in date:
        date,ms=date.split('.')
    else:
        ms=0
    tz = pytz.timezone('Etc/GMT+4')
    timeArray = time.strptime(date, "%Y-%m-%dT%H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp=timestamp.timestamp()
    timeStamp = timestamp*1000+int(ms)
    return int(timeStamp)


def timestamp_to_datetime_US(ns):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    ms=ns%1000
    ns/=1000
    dt = pytz.datetime.datetime.fromtimestamp(int(ns), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s+='.'+str(ms)
#     s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

pid_split_symble="#_"

host_split_symble="_@"


# # Database setting (Make sure the database and tables are created)

# In[3]:


import psycopg2

from psycopg2 import extras as ex
connect = psycopg2.connect(database = 'optc_db',
                           host = '/var/run/postgresql/',
                           user = 'postgres',
                           password = 'postgres',
                           port = '5432'
                          )

cur = connect.cursor()


# In[3]:


# Clear all data in the database. Run it carefully!


# In[161]:


tt=cur.execute("""
    delete from event_table where 1=1;
""")
print(tt)
connect.commit()


# In[162]:


tt=cur.execute("""
    delete from nodeid2msg where 1=1;
""")
print(tt)
connect.commit()


# ## Parse data

# In[4]:


reverse_edge_type=[
    "READ",
]



node_type_used=[
    'FILE',
 'FLOW',
 'PROCESS',
#  'SHELL',
]

def process_raw_dic(raw_dic):
    ans_dic={}
    
    
    ans_dic['hostname']=raw_dic['hostname'].split('.')[0]
    
    ans_dic['edge_type']=raw_dic['action']
    ans_dic['src_id']=raw_dic['actorID']
    ans_dic['dst_id']=raw_dic['objectID']
    
    ans_dic['src_type']='PROCESS'
    ans_dic['timestamp']=datetime_to_timestamp_US(raw_dic['timestamp'])
    ans_dic['dst_type']=raw_dic['object']
    
    try:
        node_uuid2path[ans_dic['src_id']]=ans_dic['hostname']+host_split_symble+raw_dic['properties']['image_path']  
        
    
        if raw_dic['object']=='FLOW':
            temp_flow=f"{raw_dic['properties']['direction']}#{raw_dic['properties']['src_ip']}:{raw_dic['properties']['src_port']}->{raw_dic['properties']['dest_ip']}:{raw_dic['properties']['dest_port']}"
            node_uuid2path[ans_dic['dst_id']]=ans_dic['hostname']+host_split_symble+temp_flow

        if raw_dic['object']=='FILE':              
            node_uuid2path[ans_dic['dst_id']]=ans_dic['hostname']+host_split_symble+raw_dic['properties']['file_path']


    except:
        ans_dic={}
    
    return ans_dic


# In[58]:


node_type={'FILE',
 'FLOW',
 'MODULE',
 'PROCESS',
 'REGISTRY',
 'SHELL',
 'TASK',
 'THREAD',
 'USER_SESSION'}


# # Unzip data

# In[51]:


from os import walk
 
# folder path
dir_path = '/home/monk/datasets/OpTC_data/'
 
# list to store files name
res = []
for (dir_path, dir_names, file_names) in walk(dir_path):
    if dir_path[-1]!='/':
        dir_path+='/'
#     print(f"{dir_path=}")
#     print(f"{file_names=}")
    for f in file_names:
        temp_file_path=dir_path+f
#         print(f"{temp_file_path=}")
     
        res.append(temp_file_path)


# In[10]:


for r in tqdm(res):
    if ("201-225" in r or "401-425" in r or "651-675" in r or "501-525" in r or "51-75" in r) and ".gz" in r:
        os.system(f"gzip -d {r}")
        print(f" {r} Finished！")


# # Process the features of nodes and edges

# ## Edge features

# In[16]:


# edge types
edge_set=['OPEN',
'READ',
'CREATE',
'MESSAGE',
'MODIFY',
'START',
'RENAME',
'DELETE',
'TERMINATE',
'WRITE',]

# Generate edge type one-hot
edgevec=torch.nn.functional.one_hot(torch.arange(0, len(edge_set)), num_classes=len(edge_set))


edge2vec={}
for e in range(len(edge_set)):
    edge2vec[edge_set[e]]=edgevec[e]


# In[17]:


edge2vec


# In[18]:


rel2id={}
index=1
for i in edge_set:
    rel2id[index]=i
    rel2id[i]=index
    index+=1


# In[19]:


rel2id


# ## Node features

# In[20]:


from sklearn.feature_extraction import FeatureHasher
from torch_geometric.transforms import NormalizeFeatures

from sklearn import preprocessing
import numpy as np


encode_len=16

FH_string=FeatureHasher(n_features=encode_len,input_type="string")
FH_dict=FeatureHasher(n_features=encode_len,input_type="dict")


def path2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
#     print(l)
    return l

def ip2higlist(p):
    l=[]
    if "::" not in p:
        spl=p.strip().split('.')
        for i in spl:
            if len(l)!=0:
                l.append(l[-1]+'.'+i)
            else:
                l.append(i)
    #     print(l)
        return l
    else:
        spl=p.strip().split(':')
        for i in spl:
            if len(l)!=0:
                l.append(l[-1]+':'+i)
            else:
                l.append(i)
    #     print(l)
        return l
def list2str(l):
    s=''
    for i in l:
        s+=i
    return s

def str2tensor(msg_type,msg):
    if msg_type == 'FLOW':
        h_msg=list2str(ip2higlist(msg))
    else:
        h_msg=list2str(path2higlist(msg))
    vec=FH_string.transform([msg_type+h_msg]).toarray()
    vec=torch.tensor(vec).reshape(encode_len).float()
#     print(h_msg)
    return vec


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        return self.lin(t.view(-1, 1)).cos()
    
time_enc=TimeEncoder(50)


# # Store the benign data to database

# In[141]:


node_uuid2path={}


# In[163]:


from os import walk
 
# folder path
dir_path = '/home/monk/datasets/OpTC_data/ecar/benign/'

res = []
for (dir_path, dir_names, file_names) in walk(dir_path):
    if dir_path[-1]!='/':
        dir_path+='/'
#     print(f"{dir_path=}")
#     print(f"{file_names=}")
    for f in file_names:
        temp_file_path=dir_path+f
#         print(f"{temp_file_path=}")
        if "201-225" in temp_file_path or ("20-23Sep19" in temp_file_path and ("401-425" in temp_file_path or "651-675" in temp_file_path or "501-525" in temp_file_path or "51-75" in temp_file_path)):
            res.append(temp_file_path)


# In[165]:


for r in tqdm(res):
    if  ".gz" in r:
        os.system(f"gzip -d {r}")
        print(f" {r} Finished！")


# In[166]:


def is_selected_hosts(line):
    hosts=[
        'SysClient0201',
        'SysClient0402',
        'SysClient0660',
        'SysClient0501',
        'SysClient0051',        
        'SysClient0209',
    ]
    flag=False
    for h in hosts:
        if h in line:
            flag=True
            break
    return flag


# In[167]:


for file_path in res:
    
    edge_list=[]

    with open(file_path) as f:
        for line in tqdm(f):
            line=line.replace('\\\\','/')
            temp_dic=json.loads(line.strip())
            hostname=temp_dic['hostname'].split('.')[0]
            if temp_dic['object'] in node_type_used and is_selected_hosts(hostname):
                edge_list.append(process_raw_dic(temp_dic))
    
        print(f'{len(edge_list)=}')
        data_list=[]
        for e in edge_list:
            try:
                data_list.append([
                    e['src_id'],
                    e['src_type'],
                    e['edge_type'],
                    e['dst_id'],
                    e['dst_type'],
                    e['hostname'],
                    e['timestamp'],
                    "benign",
                ])
            except:
                pass

        # write to database
        sql = '''insert into event_table
                             values %s
                '''
        ex.execute_values(cur,sql, data_list,page_size=10000)
        connect.commit()
        
        print(f"{file_path} Finished! ")
        # Clear the tmp variables to release the memory.
        del edge_list
        del data_list


# # Store the evaluation data to database

# In[168]:


from os import walk
 
# folder path
dir_path = '/home/monk/datasets/OpTC_data/ecar/evaluation/'

res = []
for (dir_path, dir_names, file_names) in walk(dir_path):
    if dir_path[-1]!='/':
        dir_path+='/'
    for f in file_names:
        temp_file_path=dir_path+f
#         print(f"{temp_file_path=}")
        if ("201-225" in temp_file_path or "401-425" in temp_file_path or "651-675" in temp_file_path or "501-525" in temp_file_path or "51-75" in temp_file_path):
            res.append(temp_file_path)


# In[170]:


for r in tqdm(res):
    if  ".gz" in r:
        os.system(f"gzip -d {r}")
        print(f" {r} Finished！")


# In[171]:


def is_selected_hosts(line):
    hosts=[
        'SysClient0201',
        'SysClient0402',
        'SysClient0660',
        'SysClient0501',
        'SysClient0051',        
        'SysClient0207',
    ]
    flag=False
    for h in hosts:
        if h in line:
            flag=True
            break
    return flag


# In[172]:


for file_path in res:
    
    edge_list=[]

    with open(file_path) as f:
        for line in tqdm(f):
            line=line.replace('\\\\','/')
            temp_dic=json.loads(line.strip())
            hostname=temp_dic['hostname'].split('.')[0]
            if temp_dic['object'] in node_type_used and is_selected_hosts(hostname):
                edge_list.append(process_raw_dic(temp_dic))
    
        print(f'{len(edge_list)=}')
        data_list=[]
        for e in edge_list:
            try:
                data_list.append([
                    e['src_id'],
                    e['src_type'],
                    e['edge_type'],
                    e['dst_id'],
                    e['dst_type'],
                    e['hostname'],
                    e['timestamp'],
                    "evaluation",
                ])
            except:
                pass

        sql = '''insert into event_table
                             values %s
                '''
        ex.execute_values(cur,sql, data_list,page_size=10000)
        connect.commit()
        
        print(f"{file_path} Finished! ")
        # Clear the tmp variables to release the memory.
        del edge_list
        del data_list


# # Store the node data into database

# In[173]:


data_list=[]
for n in node_uuid2path:
    try:
        data_list.append([
            n,
             node_uuid2path[n]
        ])
    except:
        pass
    

sql = '''insert into nodeid2msg
                     values %s
        '''
ex.execute_values(cur,sql, data_list,page_size=10000)
connect.commit()


# In[174]:


len(node_uuid2path)


# # Load node data from database

# In[12]:


# Construct the map between nodeid and msg
sql="select * from nodeid2msg;"
cur.execute(sql)
rows = cur.fetchall()

node_uuid2path={}  # nodeid => msg      node hash => nodeid
for i in tqdm(rows):
    node_uuid2path[i[0]]=i[1]


# # Generate the benign datasets

# ## h402  22

# In[54]:


for day in tqdm(range(22,23)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0402'
    datalabel='benign'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# ## h660 22

# In[55]:


for day in tqdm(range(22,23)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0660'
    datalabel='benign'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# In[ ]:





# ## h501 21

# In[7]:


for day in tqdm(range(21,22)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0501'
    datalabel='benign'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# ## h501 22

# In[57]:


for day in tqdm(range(22,23)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0501'
    datalabel='benign'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# ## h051 22

# In[59]:


for day in tqdm(range(22,23)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0051'
    datalabel='benign'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# In[ ]:





# ## h209 22

# In[60]:


for day in tqdm(range(22,23)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0209'
    datalabel='benign'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# In[ ]:





# # Generate the validation set

# ## h209 23

# In[21]:


for day in tqdm(range(23,24)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0209'
    datalabel='benign'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# In[ ]:





# # Generate the evaluation set

# ## h201 23-25

# In[61]:


for day in tqdm(range(23,26)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0201'
    datalabel='evaluation'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# ## h402 23-25

# In[62]:


for day in tqdm(range(23,26)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0402'
    datalabel='evaluation'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# ## h660 23-25

# In[63]:


for day in tqdm(range(23,26)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0660'
    datalabel='evaluation'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# ## h501 23-25

# In[64]:


for day in tqdm(range(23,26)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0501'
    datalabel='evaluation'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# In[ ]:





# ## h051 23-25

# In[65]:


for day in tqdm(range(23,26)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0051'
    datalabel='evaluation'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# In[ ]:





# ## h207 23-25

# In[66]:


for day in tqdm(range(23,26)):
    start_timestamp=datetime_to_timestamp_US('2019-09-'+str(day)+'T00:00:00')
    end_timestamp=datetime_to_timestamp_US('2019-09-'+str(day+1)+'T00:00:00')
    hostname='SysClient0207'
    datalabel='evaluation'
    sql=f"""
    select * from event_table
    where
          timestamp>{start_timestamp} and timestamp<{end_timestamp}
          and hostname='{hostname}' and data_label='{datalabel}' ORDER BY timestamp;
    """
    cur.execute(sql)
    events = cur.fetchall()
    print(f"{len(events)=}")
    
    
    
    node_set=set()
    node_uuid2index={}
    temp_index=0
    for e in events:
        if e[3] not in node_uuid2path or e[0]  not in node_uuid2path:
            continue

        if e[0] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[0]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[0]]
            temp_index+=1

        if e[3] in node_uuid2index:
            pass
        else:
            node_uuid2index[e[3]]=temp_index
            node_uuid2index[temp_index]=node_uuid2path[e[3]]
            temp_index+=1 

    torch.save(node_uuid2index,f'node_uuid2index_9_{day}_host={hostname}_datalabel={datalabel}')
       

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for e in (events):
        if e[3] in node_uuid2index and e[0] in node_uuid2index:
            # If the image path of the node is not recorded, then skip this edge
            src.append(node_uuid2index[e[0]])
            dst.append(node_uuid2index[e[3]])
        #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))

            msg.append(torch.cat([str2tensor(e[1],node_uuid2path[e[0]]), 
                                  edge2vec[e[2]], 
                                  str2tensor(e[4],node_uuid2path[e[3]])
                                 ]))
            t.append(int(e[6]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, f"./data/evaluation/9_{day}_host={hostname}_datalabel={datalabel}.TemporalData")  
    


# In[ ]:





# In[ ]:





# # A CSV file containing the ground truth nodes&edges

# In[6]:


label_df=pd.read_csv("./labels.csv")


# In[9]:


label_df


# In[48]:


nodes_attack={}
edges_attack_list=[]

for idx,row in label_df.iterrows():
    flag=False
    if row['objectID'] in node_uuid2path:
        nodes_attack[row['objectID']]=node_uuid2path[row['objectID']]
        flag=True
    if row['actorID'] in node_uuid2path:
        nodes_attack[row['actorID']]=node_uuid2path[row['actorID']]
        flag=True
    if flag and row['action'] in edge2vec:    
#         and row['action'] in edge2vec
        temp_dic={}
        temp_dic['src_uuid']=row['actorID']
        temp_dic['dst_uuid']=row['objectID']
        temp_dic['edge_type']=row['action']
        temp_dic['timestamp']=datetime_to_timestamp_US(row['timestamp'])

        edges_attack_list.append(temp_dic)


# In[49]:


len(edges_attack_list)


# In[50]:


len(nodes_attack)


# # Statistics (Num of nodes and edges)

# In[59]:


graph_9_22_h201=torch.load("./data/evaluation/9_22_host=SysClient0201_datalabel=benign.TemporalData")
graph_9_22_h402=torch.load("./data/evaluation/9_22_host=SysClient0402_datalabel=benign.TemporalData")
graph_9_22_h660=torch.load("./data/evaluation/9_22_host=SysClient0660_datalabel=benign.TemporalData")
graph_9_22_h501=torch.load("./data/evaluation/9_22_host=SysClient0501_datalabel=benign.TemporalData")
graph_9_22_h051=torch.load("./data/evaluation/9_22_host=SysClient0051_datalabel=benign.TemporalData")
graph_9_22_h209=torch.load("./data/evaluation/9_22_host=SysClient0209_datalabel=benign.TemporalData")


# In[53]:


graph_9_23_h201=torch.load("./data/evaluation/9_23_host=SysClient0201_datalabel=evaluation.TemporalData")
graph_9_24_h201=torch.load("./data/evaluation/9_24_host=SysClient0201_datalabel=evaluation.TemporalData")
graph_9_25_h201=torch.load("./data/evaluation/9_25_host=SysClient0201_datalabel=evaluation.TemporalData")


# In[54]:


graph_9_23_h402=torch.load("./data/evaluation/9_23_host=SysClient0402_datalabel=evaluation.TemporalData")
graph_9_24_h402=torch.load("./data/evaluation/9_24_host=SysClient0402_datalabel=evaluation.TemporalData")
graph_9_25_h402=torch.load("./data/evaluation/9_25_host=SysClient0402_datalabel=evaluation.TemporalData")


# In[55]:


graph_9_23_h660=torch.load("./data/evaluation/9_23_host=SysClient0660_datalabel=evaluation.TemporalData")
graph_9_24_h660=torch.load("./data/evaluation/9_24_host=SysClient0660_datalabel=evaluation.TemporalData")
graph_9_25_h660=torch.load("./data/evaluation/9_25_host=SysClient0660_datalabel=evaluation.TemporalData")


# In[56]:


graph_9_23_h501=torch.load("./data/evaluation/9_23_host=SysClient0501_datalabel=evaluation.TemporalData")
graph_9_24_h501=torch.load("./data/evaluation/9_24_host=SysClient0501_datalabel=evaluation.TemporalData")
graph_9_25_h501=torch.load("./data/evaluation/9_25_host=SysClient0501_datalabel=evaluation.TemporalData")


# In[57]:


graph_9_23_h051=torch.load("./data/evaluation/9_23_host=SysClient0051_datalabel=evaluation.TemporalData")
graph_9_24_h051=torch.load("./data/evaluation/9_24_host=SysClient0051_datalabel=evaluation.TemporalData")
graph_9_25_h051=torch.load("./data/evaluation/9_25_host=SysClient0051_datalabel=evaluation.TemporalData")


# In[58]:


graph_9_23_h207=torch.load("./data/evaluation/9_23_host=SysClient0207_datalabel=evaluation.TemporalData")
graph_9_24_h207=torch.load("./data/evaluation/9_24_host=SysClient0207_datalabel=evaluation.TemporalData")
graph_9_25_h207=torch.load("./data/evaluation/9_25_host=SysClient0207_datalabel=evaluation.TemporalData")


# In[ ]:





# In[60]:


graphs=[
    graph_9_22_h201,
    graph_9_22_h402,
    graph_9_22_h660,
    graph_9_22_h501,
    graph_9_22_h051,
    graph_9_22_h209,
    
    graph_9_23_h201,
    graph_9_24_h201,
    graph_9_25_h201,
    
    graph_9_23_h402,
    graph_9_24_h402,
    graph_9_25_h402,
    
    graph_9_23_h660,
    graph_9_24_h660,
    graph_9_25_h660,
    
    graph_9_23_h501,
    graph_9_24_h501,
    graph_9_25_h501,
    
    graph_9_23_h051,
    graph_9_24_h051,
    graph_9_25_h051,
    
    graph_9_23_h207,
    graph_9_24_h207,
    graph_9_25_h207,
]


# In[71]:


edges_count=0
for g in graphs:
     edges_count+=len(g.t)


# In[72]:


edges_count


# In[77]:


node_uuid2index_9_22_h201=torch.load("node_uuid2index_9_22_host=SysClient0201_datalabel=benign")
node_uuid2index_9_22_h402=torch.load("node_uuid2index_9_22_host=SysClient0402_datalabel=benign")
node_uuid2index_9_22_h660=torch.load("node_uuid2index_9_22_host=SysClient0660_datalabel=benign")
node_uuid2index_9_22_h501=torch.load("node_uuid2index_9_22_host=SysClient0501_datalabel=benign")
node_uuid2index_9_22_h051=torch.load("node_uuid2index_9_22_host=SysClient0051_datalabel=benign")
node_uuid2index_9_22_h209=torch.load("node_uuid2index_9_22_host=SysClient0209_datalabel=benign")


node_uuid2index_9_23_h201=torch.load("node_uuid2index_9_23_host=SysClient0201_datalabel=evaluation")
node_uuid2index_9_24_h201=torch.load("node_uuid2index_9_24_host=SysClient0201_datalabel=evaluation")
node_uuid2index_9_25_h201=torch.load("node_uuid2index_9_25_host=SysClient0201_datalabel=evaluation")

node_uuid2index_9_23_h402=torch.load("node_uuid2index_9_23_host=SysClient0402_datalabel=evaluation")
node_uuid2index_9_24_h402=torch.load("node_uuid2index_9_24_host=SysClient0402_datalabel=evaluation")
node_uuid2index_9_25_h402=torch.load("node_uuid2index_9_25_host=SysClient0402_datalabel=evaluation")

node_uuid2index_9_23_h660=torch.load("node_uuid2index_9_23_host=SysClient0660_datalabel=evaluation")
node_uuid2index_9_24_h660=torch.load("node_uuid2index_9_24_host=SysClient0660_datalabel=evaluation")
node_uuid2index_9_25_h660=torch.load("node_uuid2index_9_25_host=SysClient0660_datalabel=evaluation")

node_uuid2index_9_23_h501=torch.load("node_uuid2index_9_23_host=SysClient0501_datalabel=evaluation")
node_uuid2index_9_24_h501=torch.load("node_uuid2index_9_24_host=SysClient0501_datalabel=evaluation")
node_uuid2index_9_25_h501=torch.load("node_uuid2index_9_25_host=SysClient0501_datalabel=evaluation")

node_uuid2index_9_23_h051=torch.load("node_uuid2index_9_23_host=SysClient0051_datalabel=evaluation")
node_uuid2index_9_24_h051=torch.load("node_uuid2index_9_24_host=SysClient0051_datalabel=evaluation")
node_uuid2index_9_25_h051=torch.load("node_uuid2index_9_25_host=SysClient0051_datalabel=evaluation")

node_uuid2index_9_23_h207=torch.load("node_uuid2index_9_23_host=SysClient0207_datalabel=evaluation")
node_uuid2index_9_24_h207=torch.load("node_uuid2index_9_24_host=SysClient0207_datalabel=evaluation")
node_uuid2index_9_25_h207=torch.load("node_uuid2index_9_25_host=SysClient0207_datalabel=evaluation")





# In[78]:


node_dics=[
    node_uuid2index_9_22_h201,
    node_uuid2index_9_22_h402,
    node_uuid2index_9_22_h660,
    node_uuid2index_9_22_h501,
    node_uuid2index_9_22_h051,
    node_uuid2index_9_22_h209,
    node_uuid2index_9_23_h201,
    node_uuid2index_9_24_h201,
    node_uuid2index_9_25_h201,
    node_uuid2index_9_23_h402,
    node_uuid2index_9_24_h402,
    node_uuid2index_9_25_h402,
    node_uuid2index_9_23_h660,
    node_uuid2index_9_24_h660,
    node_uuid2index_9_25_h660,
    node_uuid2index_9_23_h501,
    node_uuid2index_9_24_h501,
    node_uuid2index_9_25_h501,
    node_uuid2index_9_23_h051,
    node_uuid2index_9_24_h051,
    node_uuid2index_9_25_h051,
    node_uuid2index_9_23_h207,
    node_uuid2index_9_24_h207,
    node_uuid2index_9_25_h207,
]


# In[86]:


nodes=set()
for dic in node_dics:
    for n in dic:
        if type(n)==str:
            nodes.add(n)


# In[87]:


len(nodes)

