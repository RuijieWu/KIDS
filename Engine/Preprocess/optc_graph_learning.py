#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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



import xxhash

def tensor_find(t,x):
    t_np=t.numpy()
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


def cal_pos_edges_loss(link_pred_ratio):
    loss=[]
    for i in link_pred_ratio:
        loss.append(criterion(i,torch.ones(1)))
    return torch.tensor(loss)

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
    loss=[] 
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)


# # Connect to database

# In[ ]:


import psycopg2

from psycopg2 import extras as ex
connect = psycopg2.connect(database = 'optc_db',
                           host = '/var/run/postgresql/',
                           user = 'postgres',
                           password = 'postgres',
                           port = '5432'
                          )

cur = connect.cursor()


# # Load data
training data
# In[ ]:


graph_9_22_h201=torch.load("./data/evaluation/9_22_host=SysClient0201_datalabel=benign.TemporalData")
graph_9_22_h402=torch.load("./data/evaluation/9_22_host=SysClient0402_datalabel=benign.TemporalData")
graph_9_22_h660=torch.load("./data/evaluation/9_22_host=SysClient0660_datalabel=benign.TemporalData")
graph_9_22_h501=torch.load("./data/evaluation/9_22_host=SysClient0501_datalabel=benign.TemporalData")
graph_9_22_h051=torch.load("./data/evaluation/9_22_host=SysClient0051_datalabel=benign.TemporalData")
graph_9_22_h209=torch.load("./data/evaluation/9_22_host=SysClient0209_datalabel=benign.TemporalData")


# In[ ]:


train_data=graph_9_22_h660


# ## Testing data

# In[ ]:


# 

graph_9_23_h201=torch.load("./data/evaluation/9_23_host=SysClient0201_datalabel=evaluation.TemporalData")
graph_9_23_h402=torch.load("./data/evaluation/9_23_host=SysClient0402_datalabel=evaluation.TemporalData")
graph_9_23_h660=torch.load("./data/evaluation/9_23_host=SysClient0660_datalabel=evaluation.TemporalData")
graph_9_23_h501=torch.load("./data/evaluation/9_23_host=SysClient0501_datalabel=evaluation.TemporalData")
graph_9_23_h051=torch.load("./data/evaluation/9_23_host=SysClient0051_datalabel=evaluation.TemporalData")
graph_9_23_h207=torch.load("./data/evaluation/9_23_host=SysClient0207_datalabel=evaluation.TemporalData")


graph_9_24_h201=torch.load("./data/evaluation/9_24_host=SysClient0201_datalabel=evaluation.TemporalData")
graph_9_24_h402=torch.load("./data/evaluation/9_24_host=SysClient0402_datalabel=evaluation.TemporalData")
graph_9_24_h660=torch.load("./data/evaluation/9_24_host=SysClient0660_datalabel=evaluation.TemporalData")
graph_9_24_h501=torch.load("./data/evaluation/9_24_host=SysClient0501_datalabel=evaluation.TemporalData")
graph_9_24_h051=torch.load("./data/evaluation/9_24_host=SysClient0051_datalabel=evaluation.TemporalData")
graph_9_24_h207=torch.load("./data/evaluation/9_24_host=SysClient0207_datalabel=evaluation.TemporalData")


graph_9_25_h201=torch.load("./data/evaluation/9_25_host=SysClient0201_datalabel=evaluation.TemporalData")
graph_9_25_h402=torch.load("./data/evaluation/9_25_host=SysClient0402_datalabel=evaluation.TemporalData")
graph_9_25_h660=torch.load("./data/evaluation/9_25_host=SysClient0660_datalabel=evaluation.TemporalData")
graph_9_25_h501=torch.load("./data/evaluation/9_25_host=SysClient0501_datalabel=evaluation.TemporalData")
graph_9_25_h051=torch.load("./data/evaluation/9_25_host=SysClient0051_datalabel=evaluation.TemporalData")
graph_9_25_h207=torch.load("./data/evaluation/9_25_host=SysClient0207_datalabel=evaluation.TemporalData")



# In[ ]:


train_data=graph_9_25_h207


# In[ ]:


graph_9_19=torch.load("./data/benign/9-19-h201.TemporalData")
graph_9_20=torch.load("./data/benign/9-20-h201.TemporalData")
graph_9_21=torch.load("./data/benign/9-21-h201.TemporalData")
graph_9_22=torch.load("./data/benign/9-22-h201.TemporalData")
graph_9_23=torch.load("./data/evaluation/9-23-h201.TemporalData")

train_data=graph_9_22


# # Generate node2msg

# In[ ]:


# Constructing the map for nodeid to msg
sql="select * from nodeid2msg;"
cur.execute(sql)
rows = cur.fetchall()

node_uuid2path={}  # nodeid => msg and node hash => nodeid
for i in tqdm(rows):
    node_uuid2path[i[0]]=i[1]


# In[ ]:


# The node2index of training data
node_uuid2index_9_17_h201=torch.load("node_uuid2index_9_17_host=SysClient0201_datalabel=benign")
node_uuid2index_9_18_h201=torch.load("node_uuid2index_9_18_host=SysClient0201_datalabel=benign")
node_uuid2index_9_19_h201=torch.load("node_uuid2index_9_19_host=SysClient0201_datalabel=benign")
node_uuid2index_9_20_h201=torch.load("node_uuid2index_9_20_host=SysClient0201_datalabel=benign")
node_uuid2index_9_21_h201=torch.load("node_uuid2index_9_21_host=SysClient0201_datalabel=benign")
node_uuid2index_9_22_h201=torch.load("node_uuid2index_9_22_host=SysClient0201_datalabel=benign")
node_uuid2index_9_23_h201=torch.load("node_uuid2index_9_23_host=SysClient0201_datalabel=benign")


# In[ ]:


node_uuid2index_9_22_h201=torch.load("node_uuid2index_9_22_host=SysClient0201_datalabel=benign")
node_uuid2index_9_22_h402=torch.load("node_uuid2index_9_22_host=SysClient0402_datalabel=benign")
node_uuid2index_9_22_h660=torch.load("node_uuid2index_9_22_host=SysClient0660_datalabel=benign")
node_uuid2index_9_22_h501=torch.load("node_uuid2index_9_22_host=SysClient0501_datalabel=benign")
node_uuid2index_9_22_h051=torch.load("node_uuid2index_9_22_host=SysClient0051_datalabel=benign")
node_uuid2index_9_22_h209=torch.load("node_uuid2index_9_22_host=SysClient0209_datalabel=benign")


# In[ ]:


# The node2index of testing data
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


# In[ ]:


maxnode_num=max(
    len(node_uuid2index_9_17_h201)//2+1,
    len(node_uuid2index_9_18_h201)//2+1,
    len(node_uuid2index_9_19_h201)//2+1,
    len(node_uuid2index_9_20_h201)//2+1,
    len(node_uuid2index_9_21_h201)//2+1,
    len(node_uuid2index_9_22_h201)//2+1,
    len(node_uuid2index_9_23_h201)//2+1,
    
    len(node_uuid2index_9_22_h201)//2+1,
    len(node_uuid2index_9_22_h402)//2+1,
    len(node_uuid2index_9_22_h660)//2+1,
    len(node_uuid2index_9_22_h501)//2+1,
    len(node_uuid2index_9_22_h051)//2+1,
    len(node_uuid2index_9_22_h209)//2+1,
    
    len(node_uuid2index_9_23_h201)//2+1,
    len(node_uuid2index_9_24_h201)//2+1,
    len(node_uuid2index_9_25_h201)//2+1,
    len(node_uuid2index_9_23_h402)//2+1,
    len(node_uuid2index_9_24_h402)//2+1,
    len(node_uuid2index_9_25_h402)//2+1,    
    
    len(node_uuid2index_9_23_h660)//2+1,
    len(node_uuid2index_9_24_h660)//2+1,
    len(node_uuid2index_9_25_h660)//2+1,
    
    len(node_uuid2index_9_23_h501)//2+1,
    len(node_uuid2index_9_24_h501)//2+1,
    len(node_uuid2index_9_25_h501)//2+1,
    
    len(node_uuid2index_9_23_h051)//2+1,
    len(node_uuid2index_9_24_h051)//2+1,
    len(node_uuid2index_9_25_h051)//2+1,
    
    len(node_uuid2index_9_23_h207)//2+1,
    len(node_uuid2index_9_24_h207)//2+1,
    len(node_uuid2index_9_25_h207)//2+1,    
)


# In[ ]:


maxnode_num


# In[ ]:


rel2id={1: 'OPEN',
 'OPEN': 1,
 2: 'READ',
 'READ': 2,
 3: 'CREATE',
 'CREATE': 3,
 4: 'MESSAGE',
 'MESSAGE': 4,
 5: 'MODIFY',
 'MODIFY': 5,
 6: 'START',
 'START': 6,
 7: 'RENAME',
 'RENAME': 7,
 8: 'DELETE',
 'DELETE': 8,
 9: 'TERMINATE',
 'TERMINATE': 9,
 10: 'WRITE',
 'WRITE': 10}


# # Setting the parameters and Training

# In[ ]:


time_dim = 100
edge_embedding_dimension=100
embedding_dim = edge_embedding_dimension
neighbor_size=20
memory_dim = 100

max_node_num = maxnode_num+2
min_dst_idx, max_dst_idx = 0, max_node_num
neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)


# In[ ]:


BATCH=1024
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels*8, out_channels,heads=1, concat=False,
                             dropout=0.0, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        last_update.to(device)
        x = x.to(device)
        t = t.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels*2)
        self.lin_dst = Linear(in_channels, in_channels*2)
        self.lin_seq = nn.Sequential(
            Linear(in_channels * 4, in_channels * 8),
            torch.nn.BatchNorm1d(in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.BatchNorm1d(in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.BatchNorm1d(int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), train_data.msg.shape[1] - 32)
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src) , self.lin_dst(z_dst)],dim=-1)      
        h = self.lin_seq (h)        
        return h

memory = TGNMemory(
    max_node_num,
    train_data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(train_data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=train_data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.00005, eps=1e-08,weight_decay=0.01)

criterion = nn.CrossEntropyLoss()


assoc = torch.empty(max_node_num, dtype=torch.long, device=device)


saved_nodes=set()

BATCH=1024
def train(train_data):


    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    saved_nodes=set()

    total_loss = 0

    for batch in train_data.seq_batches(batch_size=BATCH):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg        

        n_id = torch.cat([src, pos_dst]).unique()

        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)

        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])       

        y_pred = torch.cat([pos_out], dim=0)

        y_true=[]
        for m in msg:
            l=tensor_find(m[16:-16],1)-1
            y_true.append(l)           

        y_true = torch.tensor(y_true)
        y_true=y_true.reshape(-1).to(torch.long)

        loss = criterion(y_pred, y_true)     

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)       

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
    return total_loss / train_data.num_events



# In[ ]:


train_graphs=[
    graph_9_22_h201,
    graph_9_22_h402,
    graph_9_22_h660,
    graph_9_22_h501,
    graph_9_22_h051,
    graph_9_22_h209,
]
print(f"{embedding_dim=}")
print(f"{gnn=}")
for epoch in tqdm(range(1, 11)):
    for g in train_graphs:
        loss = train(g)
        print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')


memory.reset_state()  # Start with a fresxh memory.
neighbor_loader.reset_state() 
model=[memory,gnn, link_pred,neighbor_loader]
torch.save(model,f"./models/model_saved_traindata=hosts_9_22.pt")


# # Test

# ## Define the function for testing

# In[ ]:


import time 

@torch.no_grad()
def test_day_new(inference_data,path,nodeuuid2index):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
    memory.eval()
    gnn.eval()
    link_pred.eval()
    
    memory.reset_state()  
    neighbor_loader.reset_state()  
    
    time_with_loss={}
    total_loss = 0    
    edge_list=[]
    
    unique_nodes=torch.tensor([])
    total_edges=0
    


    start_time=int(inference_data.t[0])
    event_count=0
    
    pos_o=[]
    
    loss_list=[]


    print("after merge:",inference_data)
    

    start = time.perf_counter()

    for batch in inference_data.seq_batches(batch_size=BATCH):
        
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes=torch.cat([unique_nodes,src,pos_dst]).unique()
        total_edges+=BATCH
        
       
        n_id = torch.cat([src, pos_dst]).unique()       
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
       
        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        
        pos_o.append(pos_out)
        y_pred = torch.cat([pos_out], dim=0)

        y_true=[]
        for m in msg:
            l=tensor_find(m[16:-16],1)-1
            y_true.append(l) 
        y_true = torch.tensor(y_true)
        y_true=y_true.reshape(-1).to(torch.long)

        loss = criterion(y_pred, y_true)

        total_loss += float(loss) * batch.num_events
     
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        

        each_edge_loss= cal_pos_edges_loss_multiclass(pos_out,y_true)
        
        for i in range(len(pos_out)):
            srcnode= int(src[i])
            dstnode=  int(pos_dst[i])
            
            srcmsg=str(nodeuuid2index[srcnode]) 
            dstmsg=str(nodeuuid2index[dstnode])
            t_var=int(t[i])
            edgeindex=tensor_find(msg[i][16:-16],1) 
            edge_type=rel2id[edgeindex]
            loss=each_edge_loss[i]    

            temp_dic={}
            temp_dic['loss']=float(loss)
            temp_dic['srcnode']=srcnode
            temp_dic['dstnode']=dstnode
            temp_dic['srcmsg']=srcmsg
            temp_dic['dstmsg']=dstmsg
            temp_dic['edge_type']=edge_type
            temp_dic['time']=t_var

            edge_list.append(temp_dic)
        
        event_count+=len(batch.src)
        if t[-1]>start_time+60000*15:

            time_interval=timestamp_to_datetime_US(start_time)+"~"+timestamp_to_datetime_US(int(t[-1]))

            end = time.perf_counter()
            time_with_loss[time_interval]={'loss':loss,
                                
                                          'nodes_count':len(unique_nodes),
                                          'total_edges':total_edges,
                                          'costed_time':(end-start)}
            
            
            log=open(path+"/"+time_interval+".txt",'w')

            
            for e in edge_list: 
                loss+=e['loss']

            loss=loss/event_count   
            print(f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Cost Time: {(end-start):.2f}s')
            edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True)  
            for e in edge_list: 
                log.write(str(e))
                log.write("\n") 
            event_count=0
            total_loss=0
            loss=0
            start_time=t[-1]
            log.close()
            edge_list.clear()
            
 
    return time_with_loss


# In[ ]:





# In[ ]:





# ## 9-22 hosts

# In[ ]:


# load the models
model=torch.load("./models/model_saved_traindata=hosts_9_22.pt")
memory,gnn, link_pred,neighbor_loader=model


# In[ ]:


ans_9_22_h201=test_day_new(graph_9_22_h201,"graph_9_22_h201",node_uuid2index_9_22_h201)


# In[ ]:


ans_9_22_h402=test_day_new(graph_9_22_h402,"graph_9_22_h402",node_uuid2index_9_22_h402)


# In[ ]:


ans_9_22_h660=test_day_new(graph_9_22_h660,"graph_9_22_h660",node_uuid2index_9_22_h660)


# In[ ]:


ans_9_22_h501=test_day_new(graph_9_22_h501,"graph_9_22_h501",node_uuid2index_9_22_h501)


# In[ ]:


ans_9_22_h051=test_day_new(graph_9_22_h051,"graph_9_22_h051",node_uuid2index_9_22_h051)


# In[ ]:


ans_9_22_h209=test_day_new(graph_9_22_h209,"graph_9_22_h209",node_uuid2index_9_22_h209)


# In[ ]:





# In[ ]:





# ## 9-23~25 hosts

# In[ ]:


# load the models
model=torch.load("./models/model_saved_traindata=hosts_9_22.pt")
memory,gnn, link_pred,neighbor_loader=model


# In[ ]:


graph_9_23_h201=test_day_new(graph_9_23_h201,"graph_9_23_h201",node_uuid2index_9_23_h201)
graph_9_23_h402=test_day_new(graph_9_23_h402,"graph_9_23_h402",node_uuid2index_9_23_h402)
graph_9_23_h660=test_day_new(graph_9_23_h660,"graph_9_23_h660",node_uuid2index_9_23_h660)
graph_9_23_h501=test_day_new(graph_9_23_h501,"graph_9_23_h501",node_uuid2index_9_23_h501)
graph_9_23_h051=test_day_new(graph_9_23_h051,"graph_9_23_h051",node_uuid2index_9_23_h051)
graph_9_23_h207=test_day_new(graph_9_23_h207,"graph_9_23_h207",node_uuid2index_9_23_h207)

graph_9_24_h201=test_day_new(graph_9_24_h201,"graph_9_24_h201",node_uuid2index_9_24_h201)
graph_9_24_h402=test_day_new(graph_9_24_h402,"graph_9_24_h402",node_uuid2index_9_24_h402)
graph_9_24_h660=test_day_new(graph_9_24_h660,"graph_9_24_h660",node_uuid2index_9_24_h660)
graph_9_24_h501=test_day_new(graph_9_24_h501,"graph_9_24_h501",node_uuid2index_9_24_h501)
graph_9_24_h051=test_day_new(graph_9_24_h051,"graph_9_24_h051",node_uuid2index_9_24_h051)
graph_9_24_h207=test_day_new(graph_9_24_h207,"graph_9_24_h207",node_uuid2index_9_24_h207)

graph_9_25_h201=test_day_new(graph_9_25_h201,"graph_9_25_h201",node_uuid2index_9_25_h201)
graph_9_25_h402=test_day_new(graph_9_25_h402,"graph_9_25_h402",node_uuid2index_9_25_h402)
graph_9_25_h660=test_day_new(graph_9_25_h660,"graph_9_25_h660",node_uuid2index_9_25_h660)
graph_9_25_h501=test_day_new(graph_9_25_h501,"graph_9_25_h501",node_uuid2index_9_25_h501)
graph_9_25_h051=test_day_new(graph_9_25_h051,"graph_9_25_h051",node_uuid2index_9_25_h051)
graph_9_25_h207=test_day_new(graph_9_25_h207,"graph_9_25_h207",node_uuid2index_9_25_h207)


# In[ ]:


torch.save(graph_9_23_h201,"./test_res/graph_9_23_h201")
torch.save(graph_9_23_h402,"./test_res/graph_9_23_h402")
torch.save(graph_9_23_h660,"./test_res/graph_9_23_h660")
torch.save(graph_9_23_h501,"./test_res/graph_9_23_h501")
torch.save(graph_9_23_h051,"./test_res/graph_9_23_h051")
torch.save(graph_9_23_h207,"./test_res/graph_9_23_h207")

torch.save(graph_9_24_h201,"./test_res/graph_9_24_h201")
torch.save(graph_9_24_h402,"./test_res/graph_9_24_h402")
torch.save(graph_9_24_h660,"./test_res/graph_9_24_h660")
torch.save(graph_9_24_h501,"./test_res/graph_9_24_h501")
torch.save(graph_9_24_h051,"./test_res/graph_9_24_h051")
torch.save(graph_9_24_h207,"./test_res/graph_9_24_h207")

torch.save(graph_9_25_h201,"./test_res/graph_9_25_h201")
torch.save(graph_9_25_h402,"./test_res/graph_9_25_h402")
torch.save(graph_9_25_h660,"./test_res/graph_9_25_h660")
torch.save(graph_9_25_h501,"./test_res/graph_9_25_h501")
torch.save(graph_9_25_h051,"./test_res/graph_9_25_h051")
torch.save(graph_9_25_h207,"./test_res/graph_9_25_h207")


# # Compute the anomalous score

# In[ ]:


def cal_train_IDF(find_str,file_list):
    include_count=0
    for f_path in (file_list):
        f=open(f_path)
        if find_str in f.read():
            include_count+=1             
    IDF=math.log(len(file_list)/(include_count+1))
    return IDF


def cal_IDF(find_str,file_path,file_list):
    file_list=os.listdir(file_path)
    include_count=0
    different_neighbor=set()
    for f_path in (file_list):
        f=open(file_path+f_path)
        if find_str in f.read():
            include_count+=1
                
    IDF=math.log(len(file_list)/(include_count+1))
    
    return IDF,1


def cal_IDF_by_file_in_mem(find_str,file_list):
    include_count=0
    different_neighbor=set()
    for f in (file_list):       
        if find_str in f:
            include_count+=1
    IDF=math.log(len(file_list)/(include_count+1))    
    return IDF

def cal_redundant(find_str,edge_list):
    
    different_neighbor=set()
    for e in edge_list:
        if find_str in str(e):
            different_neighbor.add(e[0])
            different_neighbor.add(e[1])
    return len(different_neighbor)-2

def cal_anomaly_loss(loss_list,edge_list,file_path):
    
    if len(loss_list)!=len(edge_list):
        print("error!")
        return 0
    count=0
    loss_sum=0
    loss_std=std(loss_list)
    loss_mean=mean(loss_list)
    edge_set=set()
    node_set=set()
    node2redundant={}
    
    thr=loss_mean+2.5*loss_std

    print("thr:",thr)
    
    for i in range(len(loss_list)):
        if loss_list[i]>thr:
            count+=1
            src_node=edge_list[i][0]
            dst_node=edge_list[i][1]

            loss_sum+=loss_list[i]
    
            node_set.add(src_node)
            node_set.add(dst_node)
            edge_set.add(edge_list[i][0]+edge_list[i][1])
    return count, loss_sum/count,node_set,edge_set
#     return count, count/len(loss_list)


# # Construct the relations between time windows

# In[ ]:


file_list=[]

file_path="graph_9_22_h201/"
file_l=os.listdir("graph_9_22_h201/")
for i in file_l:
    file_list.append(file_path+i)

file_path="graph_9_22_h402/"
file_l=os.listdir("graph_9_22_h402/")
for i in file_l:
    file_list.append(file_path+i)
    
file_path="graph_9_22_h660/"
file_l=os.listdir("graph_9_22_h660/")
for i in file_l:
    file_list.append(file_path+i)
    
file_path="graph_9_22_h501/"
file_l=os.listdir("graph_9_22_h501/")
for i in file_l:
    file_list.append(file_path+i)
    
file_path="graph_9_22_h051/"
file_l=os.listdir("graph_9_22_h051/")
for i in file_l:
    file_list.append(file_path+i)    
    
    
file_path="graph_9_22_h209/"
file_l=os.listdir("graph_9_22_h209/")
for i in file_l:
    file_list.append(file_path+i) 

    
def is_include_key_word(s):
    keywords=[
         '->',
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
      ]
    flag=False
    for i in keywords:
        if i in s:
            flag=True
    return flag    
    
    
def cal_set_rel(s1,s2):
    new_s=s1 & s2
    count=0
    for i in new_s:
        if is_include_key_word(i) is not True:

            if i in node_IDF.keys():
                IDF=node_IDF[i]
            else:
                IDF=math.log(len(file_list)/(1))
            if IDF>(math.log(len(file_list)*0.9)):
                print("node:",i," IDF:",IDF)
                count+=1
    return count


# In[ ]:


math.log(len(file_list)/(1))


# # Compute the IDF-1

# In[ ]:


file_list=[]

file_path="graph_9_22_h201/"
file_l=os.listdir("graph_9_22_h201/")
for i in file_l:
    file_list.append(file_path+i)

file_path="graph_9_22_h402/"
file_l=os.listdir("graph_9_22_h402/")
for i in file_l:
    file_list.append(file_path+i)
    
file_path="graph_9_22_h660/"
file_l=os.listdir("graph_9_22_h660/")
for i in file_l:
    file_list.append(file_path+i)
    
file_path="graph_9_22_h501/"
file_l=os.listdir("graph_9_22_h501/")
for i in file_l:
    file_list.append(file_path+i)
    
file_path="graph_9_22_h051/"
file_l=os.listdir("graph_9_22_h051/")
for i in file_l:
    file_list.append(file_path+i)    
    
    
file_path="graph_9_22_h209/"
file_l=os.listdir("graph_9_22_h209/")
for i in file_l:
    file_list.append(file_path+i)    



# In[ ]:


node_set=set()

for f_path in tqdm(file_list):
    f=open(f_path)
    for line in f:
        l=line.strip()
        jdata=eval(l)
        if jdata['loss']>0:
            if '->' not in str(jdata['srcmsg']):
                node_set.add(str(jdata['srcmsg']).split("_@")[-1])
            if '->' not in str(jdata['dstmsg']):
                node_set.add(str(jdata['dstmsg']).split("_@")[-1]) 


node_list=list(node_set)
del node_set


# In[ ]:


files_mem_list=[]
for f_path in (file_list):
        f=open(f_path)
        files_mem_list.append(f.read())


# In[ ]:


def process_node_idf(_node_list,_files_mem_list,share_node_IDF):
    for n in tqdm(_node_list):  
        find_str=n
        IDF=cal_IDF_by_file_in_mem(n,_files_mem_list)
        share_node_IDF[n]=IDF

import multiprocessing as mp


# In[ ]:


share_node_IDF = mp.Manager().dict()
cores=28
offset=math.ceil(len(node_list)/cores)
node_list_split=[]
for i in range(0,len(node_list),offset):
    node_list_split+=[node_list[i:i+offset]]     
process_list=[]
for i in range(cores):
    process_list.append(mp.Process(target=process_node_idf, args=(node_list_split[i],files_mem_list,share_node_IDF)))
for i in process_list:
    i.start()


# In[ ]:


node_IDF=dict(share_node_IDF)
torch.save(node_IDF,"node_IDF_9_22_hosts")
print("IDF weight calculate complete!")


# In[ ]:


# force to terminate all process
for i in process_list:
    i.terminate()


# In[ ]:


# The IDF of the date 9-22 h201
node_IDF={}
node_set=set()

file_list=[]

file_path="graph_9_22/"
file_l=os.listdir("graph_9_22/")


for i in file_l:
    file_list.append(file_path+i)

for f_path in tqdm(file_list):
    f=open(f_path)
    for line in f:
        l=line.strip()
        jdata=eval(l)
        if jdata['loss']>0:
            if '->' not in str(jdata['srcmsg']):
                node_set.add(str(jdata['srcmsg']))
            if '->' not in str(jdata['dstmsg']):
                node_set.add(str(jdata['dstmsg'])) 

for n in tqdm(node_set):
#     find_str=list(eval(n).values())[0]
    IDF=cal_train_IDF(n,file_list)
    node_IDF[n]=IDF


torch.save(node_IDF,"node_IDF_9_22_without_netflow")
print("IDF weight calculate complete!")


# In[ ]:





# In[ ]:





# In[ ]:





# # labelling

# In[ ]:


labels={}
path="graph_9_23_h201/"
filelist = os.listdir(path)
filelist.sort()
for f in filelist:
    labels[path+f]=0


# ## h201

# In[ ]:


label_h201={}

test_path="graph_9_23_h201/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h201[test_path+f]=0

test_path="graph_9_24_h201/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h201[test_path+f]=0

test_path="graph_9_25_h201/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h201[test_path+f]=0

attack_list=[
'graph_9_23_h201/2019-09-23 11:23:44.136~2019-09-23 11:38:30.698.txt',
 'graph_9_23_h201/2019-09-23 11:38:40.698~2019-09-23 11:53:39.57.txt',

 'graph_9_23_h201/2019-09-23 12:38:24.95~2019-09-23 12:54:14.286.txt',
 'graph_9_23_h201/2019-09-23 12:55:28.286~2019-09-23 13:09:50.95.txt',
 'graph_9_23_h201/2019-09-23 13:10:24.95~2019-09-23 13:24:56.43.txt',

]
for i in attack_list:
    label_h201[i]=1


# ## h402

# In[ ]:


label_h402={}

test_path="graph_9_23_h402/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h402[test_path+f]=0

test_path="graph_9_24_h402/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h402[test_path+f]=0

test_path="graph_9_25_h402/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h402[test_path+f]=0

attack_list=[
 'graph_9_23_h402/2019-09-23 13:10:24.429~2019-09-23 13:25:09.374.txt',
 'graph_9_23_h402/2019-09-23 13:25:20.374~2019-09-23 13:40:21.268.txt',

    # adjust
#      'graph_9_23_h402/2019-09-23 13:40:16.268~2019-09-23 13:55:31.31.txt',
#  'graph_9_23_h402/2019-09-23 13:55:12.31~2019-09-23 14:10:58.2.txt',
]
for i in attack_list:
    label_h402[i]=1


# ## h660

# In[ ]:


label_h660={}

test_path="graph_9_23_h660/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h660[test_path+f]=0

test_path="graph_9_24_h660/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h660[test_path+f]=0

test_path="graph_9_25_h660/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h660[test_path+f]=0

attack_list=[
'graph_9_23_h660/2019-09-23 13:27:28.512~2019-09-23 13:42:30.682.txt',
 'graph_9_23_h660/2019-09-23 13:42:24.682~2019-09-23 13:57:57.566.txt',
 'graph_9_23_h660/2019-09-23 13:57:20.566~2019-09-23 14:12:59.139.txt',
]
for i in attack_list:
    label_h660[i]=1


# In[ ]:


label_h660


# ## h501

# In[ ]:


label_h501={}

test_path="graph_9_23_h501/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h501[test_path+f]=0

test_path="graph_9_24_h501/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h501[test_path+f]=0

test_path="graph_9_25_h501/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h501[test_path+f]=0

attack_list=[

 'graph_9_24_h501/2019-09-24 10:15:28.241~2019-09-24 10:30:00.201.txt',
 'graph_9_24_h501/2019-09-24 10:30:24.201~2019-09-24 10:45:02.7.txt',
 'graph_9_24_h501/2019-09-24 10:45:20.7~2019-09-24 11:00:31.385.txt',
 'graph_9_24_h501/2019-09-24 11:00:16.385~2019-09-24 11:16:09.755.txt',
 'graph_9_24_h501/2019-09-24 11:15:12.755~2019-09-24 11:31:14.287.txt',
 'graph_9_24_h501/2019-09-24 11:32:16.287~2019-09-24 11:46:31.541.txt',

 'graph_9_24_h501/2019-09-24 13:04:00.804~2019-09-24 13:17:29.451.txt',
 'graph_9_24_h501/2019-09-24 13:18:56.451~2019-09-24 13:32:46.454.txt',
 'graph_9_24_h501/2019-09-24 13:33:52.454~2019-09-24 13:48:02.493.txt',

# adjust

#  'graph_9_24_h501/2019-09-24 11:47:12.541~2019-09-24 12:01:32.699.txt',
#  'graph_9_24_h501/2019-09-24 12:02:08.699~2019-09-24 12:16:34.904.txt',
#  'graph_9_24_h501/2019-09-24 12:32:00.112~2019-09-24 12:46:58.691.txt',
#  'graph_9_24_h501/2019-09-24 12:46:56.691~2019-09-24 13:02:02.804.txt',
#  'graph_9_24_h501/2019-09-24 16:22:24.281~2019-09-24 16:36:06.878.txt',

]
for i in attack_list:
    label_h501[i]=1


# In[ ]:


label_h501


# In[ ]:





# ## h051

# In[ ]:


label_h051={}

test_path="graph_9_23_h051/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h051[test_path+f]=0

test_path="graph_9_24_h051/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h051[test_path+f]=0

test_path="graph_9_25_h051/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h051[test_path+f]=0

attack_list=[
'graph_9_25_h051/2019-09-25 10:26:08.397~2019-09-25 10:41:40.247.txt',
 'graph_9_25_h051/2019-09-25 10:41:04.247~2019-09-25 10:56:56.92.txt',
 'graph_9_25_h051/2019-09-25 10:56:00.92~2019-09-25 11:12:03.608.txt',
]
for i in attack_list:
    label_h051[i]=1


# In[ ]:


label_h051


# In[ ]:





# ## h207

# In[ ]:


label_h207={}

test_path="graph_9_23_h207/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h207[test_path+f]=0

test_path="graph_9_24_h207/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h207[test_path+f]=0

test_path="graph_9_25_h207/"

filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    label_h207[test_path+f]=0

attack_list=[

]
for i in attack_list:
    label_h207[i]=1


# In[ ]:





# # Anoamly detection

# In[ ]:


def classifier_evaluation(y_test, y_test_pred):
    # groundtruth, pred_value
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
#     tn+=100
#     print(clf_name," : ")
    print('tn:',tn)
    print('fp:',fp)
    print('fn:',fn)
    print('tp:',tp)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)
    print("precision:",precision)
    print("recall:",recall)
    print("fscore:",fscore)
    print("accuracy:",accuracy)
    print("auc_val:",auc_val)
    return precision,recall,fscore,accuracy,auc_val


# ## h201

# ### 9-23

# In[ ]:


node_IDF=torch.load("node_IDF_9_22_hosts")

# node_set_list=[]
history_list_9_23_h201=[]
tw_que=[]
his_tw={}
current_tw={}

test_path="graph_9_23_h201/"


file_l=os.listdir(test_path)
file_l.sort()
index_count=0
for f_path in (file_l):
    f=open(test_path+f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']).split("_@")[-1],str(jdata['dstmsg']).split("_@")[-1]])
#         edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
        
#     df_list_9_22.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,test_path)
    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    added_que_flag=False
    for hq in history_list_9_23_h201:
        for his_tw in hq:
            if cal_set_rel(current_tw['nodeset'],his_tw['nodeset'])!=0 and current_tw['name']!=his_tw['name']:
                hq.append(copy.deepcopy(current_tw))
                print(f"{his_tw['name']=}")
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list_9_23_h201.append(temp_hq)
  
    index_count+=1
#     node_set_list.append(node_set)
    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))
#     y_data_4_10.append([loss_avg,labels_4_10[f_path],f_path])


# In[ ]:


pred_label_h201={}

test_path="graph_9_23_h201/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h201[test_path+f]=0
    
test_path="graph_9_24_h201/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h201[test_path+f]=0
    
test_path="graph_9_25_h201/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h201[test_path+f]=0
    
    
    
    
for hl in history_list_9_23_h201:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
    name_list=[]
    if loss_count>1000:
        name_list=[]
        for i in hl:
            name_list.append(i['name'])
        print(*name_list, sep = "\n")
        for i in name_list:
            pred_label_h201["graph_9_23_h201/"+i]=1
        print(loss_count)


# In[ ]:


# evaluate
y=[]
y_pred=[]
for i in label_h201:
    y.append(label_h201[i])
    y_pred.append(pred_label_h201[i])
    
classifier_evaluation(y,y_pred)


# ### 9-24

# In[ ]:


node_IDF=torch.load("node_IDF_9_22_hosts")

# node_set_list=[]
history_list=[]
tw_que=[]
his_tw={}
current_tw={}

test_path="graph_9_24_h201/"


file_l=os.listdir(test_path)
file_l.sort()
index_count=0
for f_path in (file_l):
    f=open(test_path+f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']).split("_@")[-1],str(jdata['dstmsg']).split("_@")[-1]])
#         edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
        
#     df_list_9_22.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,test_path)
    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    added_que_flag=False
    for hq in history_list:
        for his_tw in hq:
            if cal_set_rel(current_tw['nodeset'],his_tw['nodeset'])!=0 and current_tw['name']!=his_tw['name']:
                hq.append(copy.deepcopy(current_tw))
                print(f"{his_tw['name']=}")
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list.append(temp_hq)
  
    index_count+=1
#     node_set_list.append(node_set)
    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))
#     y_data_4_10.append([loss_avg,labels_4_10[f_path],f_path])


# In[ ]:


for hl in history_list:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
    name_list=[]
    if loss_count>10000:
        name_list=[]
        for i in hl:
            name_list.append(i['name'])
        print(name_list)
#         for i in name_list:
#             pred_label[i]=1
        print(loss_count)


# ### 9-25

# In[ ]:


node_IDF=torch.load("node_IDF_9_22_hosts")

# node_set_list=[]
history_list=[]
tw_que=[]
his_tw={}
current_tw={}

test_path="graph_9_25_h201/"


file_l=os.listdir(test_path)
file_l.sort()
index_count=0
for f_path in (file_l):
    f=open(test_path+f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']).split("_@")[-1],str(jdata['dstmsg']).split("_@")[-1]])
#         edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,test_path)
    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    added_que_flag=False
    for hq in history_list:
        for his_tw in hq:
            if cal_set_rel(current_tw['nodeset'],his_tw['nodeset'])!=0 and current_tw['name']!=his_tw['name']:
                hq.append(copy.deepcopy(current_tw))
                print(f"{his_tw['name']=}")
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list.append(temp_hq)
  
    index_count+=1

    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))


# In[ ]:


# pred_label={}

    
# filelist = os.listdir("graph_9_23")
# for f in filelist:
#     pred_label[f]=0


for hl in history_list:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
    name_list=[]
    if loss_count>10000:
        name_list=[]
        for i in hl:
            name_list.append(i['name'])
        print(name_list)
#         for i in name_list:
#             pred_label[i]=1
        print(loss_count)


# In[ ]:





# ## h402

# In[ ]:


node_IDF=torch.load("node_IDF_9_22_hosts")

# node_set_list=[]
history_list_9_23_h402=[]
tw_que=[]
his_tw={}
current_tw={}

test_path="graph_9_23_h402/"


file_l=os.listdir(test_path)
file_l.sort()
index_count=0
for f_path in (file_l):
    f=open(test_path+f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']).split("_@")[-1],str(jdata['dstmsg']).split("_@")[-1]])
#         edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
        
#     df_list_9_22.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,test_path)
    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    added_que_flag=False
    for hq in history_list_9_23_h402:
        for his_tw in hq:
            if cal_set_rel(current_tw['nodeset'],his_tw['nodeset'])!=0 and current_tw['name']!=his_tw['name']:
                hq.append(copy.deepcopy(current_tw))
                print(f"{his_tw['name']=}")
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list_9_23_h402.append(temp_hq)
  
    index_count+=1
#     node_set_list.append(node_set)
    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))
#     y_data_4_10.append([loss_avg,labels_4_10[f_path],f_path])


# In[ ]:


pred_label_h402={}

test_path="graph_9_23_h402/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h402[test_path+f]=0
    
test_path="graph_9_24_h402/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h402[test_path+f]=0
    
test_path="graph_9_25_h402/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h402[test_path+f]=0
    
    
    
    
for hl in history_list_9_23_h402:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
    name_list=[]
    if loss_count>1000:
        name_list=[]
        for i in hl:
            name_list.append(i['name'])
        print(*name_list, sep = "\n")
        for i in name_list:
            pred_label_h402["graph_9_23_h402/"+i]=1
        print(loss_count)


# In[ ]:


# Evalaute
y=[]
y_pred=[]
for i in label_h402:
    y.append(label_h402[i])
    y_pred.append(pred_label_h402[i])
    
classifier_evaluation(y,y_pred)


# In[ ]:





# ## h660

# In[ ]:


node_IDF=torch.load("node_IDF_9_22_hosts")

# node_set_list=[]
history_list_9_23_h660=[]
tw_que=[]
his_tw={}
current_tw={}

test_path="graph_9_23_h660/"


file_l=os.listdir(test_path)
file_l.sort()
index_count=0
for f_path in (file_l):
    f=open(test_path+f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']).split("_@")[-1],str(jdata['dstmsg']).split("_@")[-1]])
#         edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
        
#     df_list_9_22.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,test_path)
    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    added_que_flag=False
    for hq in history_list_9_23_h660:
        for his_tw in hq:
            if cal_set_rel(current_tw['nodeset'],his_tw['nodeset'])!=0 and current_tw['name']!=his_tw['name']:
                hq.append(copy.deepcopy(current_tw))
                print(f"{his_tw['name']=}")
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list_9_23_h660.append(temp_hq)
  
    index_count+=1
#     node_set_list.append(node_set)
    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))
#     y_data_4_10.append([loss_avg,labels_4_10[f_path],f_path])


# In[ ]:


pred_label_h660={}

test_path="graph_9_23_h660/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h660[test_path+f]=0
    
test_path="graph_9_24_h660/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h660[test_path+f]=0
    
test_path="graph_9_25_h660/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h660[test_path+f]=0
    
    
    
    
for hl in history_list_9_23_h660:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
    name_list=[]
    if loss_count>100:
        name_list=[]
        for i in hl:
            name_list.append(i['name'])
        print(*name_list, sep = "\n")
        for i in name_list:
            pred_label_h660["graph_9_23_h660/"+i]=1
        print(loss_count)
        
        
# evalute
y=[]
y_pred=[]
for i in label_h660:
    y.append(label_h660[i])
    y_pred.append(pred_label_h660[i])
    
classifier_evaluation(y,y_pred)


# ## h501

# In[ ]:


node_IDF=torch.load("node_IDF_9_22_hosts")

# node_set_list=[]
history_list_9_24_h501=[]
tw_que=[]
his_tw={}
current_tw={}

test_path="graph_9_24_h501/"


file_l=os.listdir(test_path)
file_l.sort()
index_count=0
for f_path in (file_l):
    f=open(test_path+f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']).split("_@")[-1],str(jdata['dstmsg']).split("_@")[-1]])
#         edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
        
#     df_list_9_22.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,test_path)
    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    added_que_flag=False
    for hq in history_list_9_24_h501:
        for his_tw in hq:
            if cal_set_rel(current_tw['nodeset'],his_tw['nodeset'])!=0 and current_tw['name']!=his_tw['name']:
                hq.append(copy.deepcopy(current_tw))
                print(f"{his_tw['name']=}")
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list_9_24_h501.append(temp_hq)
  
    index_count+=1
#     node_set_list.append(node_set)
    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))
#     y_data_4_10.append([loss_avg,labels_4_10[f_path],f_path])


# In[ ]:


pred_label_h501={}

test_path="graph_9_23_h501/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h501[test_path+f]=0
    
test_path="graph_9_24_h501/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h501[test_path+f]=0
    
test_path="graph_9_25_h501/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h501[test_path+f]=0
    
    
    
    
for hl in history_list_9_24_h501:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
    name_list=[]
    if loss_count>200:
        name_list=[]
        for i in hl:
            name_list.append(i['name'])
        print(*name_list, sep = "\n")
        for i in name_list:
            pred_label_h501["graph_9_24_h501/"+i]=1
        print(loss_count)
        
        
# evaluate
y=[]
y_pred=[]
for i in label_h501:
    y.append(label_h501[i])
    y_pred.append(pred_label_h501[i])
    
classifier_evaluation(y,y_pred)


# ## h051

# In[ ]:


# node_IDF=torch.load("node_IDF_4_6")
node_IDF=torch.load("node_IDF_9_22_hosts")

# node_set_list=[]
history_list_9_25_h051=[]
tw_que=[]
his_tw={}
current_tw={}

test_path="graph_9_25_h051/"


file_l=os.listdir(test_path)
file_l.sort()
index_count=0
for f_path in (file_l):
    f=open(test_path+f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']).split("_@")[-1],str(jdata['dstmsg']).split("_@")[-1]])
#         edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
        
#     df_list_9_22.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,test_path)
    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    added_que_flag=False
    for hq in history_list_9_25_h051:
        for his_tw in hq:
            if cal_set_rel(current_tw['nodeset'],his_tw['nodeset'])!=0 and current_tw['name']!=his_tw['name']:
                hq.append(copy.deepcopy(current_tw))
                print(f"{his_tw['name']=}")
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list_9_25_h051.append(temp_hq)
  
    index_count+=1
#     node_set_list.append(node_set)
    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))
#     y_data_4_10.append([loss_avg,labels_4_10[f_path],f_path])


# In[ ]:


pred_label_h051={}

test_path="graph_9_23_h051/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h051[test_path+f]=0
    
test_path="graph_9_24_h051/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h051[test_path+f]=0
    
test_path="graph_9_25_h051/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h051[test_path+f]=0
    
    
    
    
for hl in history_list_9_25_h051:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
    name_list=[]
    if loss_count>2000:
        name_list=[]
        for i in hl:
            name_list.append(i['name'])

        print(*name_list, sep = "\n")
        for i in name_list:
            pred_label_h051["graph_9_25_h051/"+i]=1
        print(loss_count)
        
        
# evaluate
y=[]
y_pred=[]
for i in label_h051:
    y.append(label_h051[i])
    y_pred.append(pred_label_h051[i])
    
classifier_evaluation(y,y_pred)


# In[ ]:





# In[ ]:





# ## h207

# In[ ]:


node_IDF=torch.load("node_IDF_9_22_hosts")

# node_set_list=[]
history_list_9_25_h207=[]
tw_que=[]
his_tw={}
current_tw={}

test_path="graph_9_25_h207/"


file_l=os.listdir(test_path)
file_l.sort()
index_count=0
for f_path in (file_l):
    f=open(test_path+f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']).split("_@")[-1],str(jdata['dstmsg']).split("_@")[-1]])
#         edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
        
#     df_list_9_22.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,test_path)
    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    added_que_flag=False
    for hq in history_list_9_25_h207:
        for his_tw in hq:
            if cal_set_rel(current_tw['nodeset'],his_tw['nodeset'])!=0 and current_tw['name']!=his_tw['name']:
                hq.append(copy.deepcopy(current_tw))
                print(f"{his_tw['name']=}")
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list_9_25_h207.append(temp_hq)
  
    index_count+=1

    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))


# In[ ]:


pred_label_h207={}

test_path="graph_9_23_h207/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h207[test_path+f]=0
    
test_path="graph_9_24_h207/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h207[test_path+f]=0
    
test_path="graph_9_25_h207/"
    
filelist = os.listdir(test_path)
filelist.sort()
for f in filelist:
    pred_label_h207[test_path+f]=0
    
    
    
    
for hl in history_list_9_25_h207:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
    name_list=[]
    if loss_count>2000:
        name_list=[]
        for i in hl:
            name_list.append(i['name'])

        print(*name_list, sep = "\n")
        for i in name_list:
            pred_label_h207["graph_9_25_h207/"+i]=1
        print(loss_count)
        
        


# In[ ]:


# evaluate
y=[]
y_pred=[]
for i in label_h207:
    y.append(label_h207[i])
    y_pred.append(pred_label_h207[i])
    
    
classifier_evaluation(y,y_pred)


# ## Overall evaluation

# In[ ]:


y=[]
y_pred=[]

for i in label_h201:
    y.append(label_h201[i])
    y_pred.append(pred_label_h201[i])

for i in label_h402:
    y.append(label_h402[i])
    y_pred.append(pred_label_h402[i])

for i in label_h660:
    y.append(label_h660[i])
    y_pred.append(pred_label_h660[i])

for i in label_h501:
    y.append(label_h501[i])
    y_pred.append(pred_label_h501[i])

for i in label_h051:
    y.append(label_h051[i])
    y_pred.append(pred_label_h051[i])

for i in label_h207:
    y.append(label_h207[i])
    y_pred.append(pred_label_h207[i])
    
    
classifier_evaluation(y,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# # Attack investigation

# In[ ]:


replace_dic={
    '.pyc':'*.pyc',
#     '.dll':'*.dll',
#     '.DLL':'*.DLL',
}

def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    if '->' in path_name:
        if 'outbound' in path_name:
            msg=re.findall("->(.*?):",path_name)[0]
            return msg
        elif 'inbound' in path_name:
            msg=re.findall("#(.*?):",path_name)[0]
            return msg
    return path_name


# In[ ]:





# In[ ]:





# ## Count the number of attack edges

# In[ ]:


label_df=pd.read_csv("./labels.csv")


# ### h201

# In[ ]:


nodes_attack_h201={}
attack_node_set_h201=set()
edges_attack_hashset_h201=set()
h201_df_label=label_df[label_df['hostname']=='SysClient0201.systemia.com']

for idx,row in h201_df_label.iterrows():
    
    srcflag=False
    dstflag=False
    if row['objectID'] in node_uuid2path:
        nodes_attack_h201[row['objectID']]=replace_path_name(node_uuid2path[row['objectID']])

        dstflag=True
    if row['actorID'] in node_uuid2path:
        nodes_attack_h201[row['actorID']]=replace_path_name(node_uuid2path[row['actorID']])
        
        srcflag=True
    if srcflag and dstflag and row['action'] in rel2id:    
        temp_edge=replace_path_name(node_uuid2path[row['actorID']])+','+replace_path_name(node_uuid2path[row['objectID']])+','+str(datetime_to_timestamp_US(row['timestamp']))
        edges_attack_hashset_h201.add(hashgen(temp_edge))

for i in nodes_attack_h201:
    attack_node_set_h201.add(nodes_attack_h201[i])


# ### h402

# In[ ]:


nodes_attack_h402={}
attack_node_set_h402=set()
edges_attack_hashset_h402=set()
h402_df_label=label_df[label_df['hostname']=='SysClient0402.systemia.com']

for idx,row in h402_df_label.iterrows():
    
    srcflag=False
    dstflag=False
    if row['objectID'] in node_uuid2path:
        nodes_attack_h402[row['objectID']]=replace_path_name(node_uuid2path[row['objectID']])
        dstflag=True
    if row['actorID'] in node_uuid2path:
        nodes_attack_h402[row['actorID']]=replace_path_name(node_uuid2path[row['actorID']])
        
        srcflag=True
    if srcflag and dstflag and row['action'] in rel2id:    
        temp_edge=replace_path_name(node_uuid2path[row['actorID']])+','+replace_path_name(node_uuid2path[row['objectID']])+','+str(datetime_to_timestamp_US(row['timestamp']))
        edges_attack_hashset_h402.add(hashgen(temp_edge))

for i in nodes_attack_h402:
    attack_node_set_h402.add(nodes_attack_h402[i])


# In[ ]:


len(edges_attack_hashset_h402)


# ### h660

# In[ ]:


nodes_attack_h660={}
attack_node_set_h660=set()
edges_attack_hashset_h660=set()
h660_df_label=label_df[label_df['hostname']=='SysClient0660.systemia.com']

for idx,row in h660_df_label.iterrows():
    
    srcflag=False
    dstflag=False
    if row['objectID'] in node_uuid2path:
        nodes_attack_h660[row['objectID']]=replace_path_name(node_uuid2path[row['objectID']])
        dstflag=True
    if row['actorID'] in node_uuid2path:
        nodes_attack_h660[row['actorID']]=replace_path_name(node_uuid2path[row['actorID']])
        srcflag=True
    if srcflag and dstflag and row['action'] in rel2id:    
        temp_edge=replace_path_name(node_uuid2path[row['actorID']])+','+replace_path_name(node_uuid2path[row['objectID']])+','+str(datetime_to_timestamp_US(row['timestamp']))
        edges_attack_hashset_h660.add(hashgen(temp_edge))

for i in nodes_attack_h660:
    attack_node_set_h660.add(nodes_attack_h660[i])


# In[ ]:


len(edges_attack_hashset_h660)


# ### h501

# In[ ]:


nodes_attack_h501={}
attack_node_set_h501=set()
edges_attack_hashset_h501=set()
h501_df_label=label_df[label_df['hostname']=='SysClient0501.systemia.com']

for idx,row in h501_df_label.iterrows():
    
    srcflag=False
    dstflag=False
    if row['objectID'] in node_uuid2path:
        nodes_attack_h501[row['objectID']]=replace_path_name(node_uuid2path[row['objectID']])
        dstflag=True
    if row['actorID'] in node_uuid2path:
        nodes_attack_h501[row['actorID']]=replace_path_name(node_uuid2path[row['actorID']])
        srcflag=True
    if srcflag and dstflag and row['action'] in rel2id:    
        temp_edge=replace_path_name(node_uuid2path[row['actorID']])+','+replace_path_name(node_uuid2path[row['objectID']])+','+str(datetime_to_timestamp_US(row['timestamp']))
        edges_attack_hashset_h501.add(hashgen(temp_edge))

for i in nodes_attack_h501:
    attack_node_set_h501.add(nodes_attack_h501[i])


# ### h051

# In[ ]:


nodes_attack_h051={}
attack_node_set_h051=set()
edges_attack_hashset_h051=set()
h051_df_label=label_df[label_df['hostname']=='SysClient0051.systemia.com']

for idx,row in h051_df_label.iterrows():
    
    srcflag=False
    dstflag=False
    if row['objectID'] in node_uuid2path:
        nodes_attack_h051[row['objectID']]=replace_path_name(node_uuid2path[row['objectID']])
        dstflag=True
    if row['actorID'] in node_uuid2path:
        nodes_attack_h051[row['actorID']]=replace_path_name(node_uuid2path[row['actorID']])
        srcflag=True
    if srcflag and dstflag and row['action'] in rel2id:    
        temp_edge=replace_path_name(node_uuid2path[row['actorID']])+','+replace_path_name(node_uuid2path[row['objectID']])+','+str(datetime_to_timestamp_US(row['timestamp']))
        edges_attack_hashset_h051.add(hashgen(temp_edge))

for i in nodes_attack_h051:
    attack_node_set_h051.add(nodes_attack_h051[i])


# In[ ]:





# In[ ]:





# ## h201

# In[ ]:


attack_list=[
    'graph_9_23_h201/2019-09-23 09:52:00.187~2019-09-23 10:06:52.508.txt',
  'graph_9_23_h201/2019-09-23 11:23:44.136~2019-09-23 11:38:30.698.txt',
 'graph_9_23_h201/2019-09-23 11:38:40.698~2019-09-23 11:53:39.57.txt',
 
 'graph_9_23_h201/2019-09-23 12:38:24.95~2019-09-23 12:54:14.286.txt',
 'graph_9_23_h201/2019-09-23 12:55:28.286~2019-09-23 13:09:50.95.txt',
 'graph_9_23_h201/2019-09-23 13:10:24.95~2019-09-23 13:24:56.43.txt',
    'graph_9_23_h201/2019-09-23 15:24:48.426~2019-09-23 15:42:02.967.txt',
            ]


# In[ ]:


from datetime import datetime, timezone
attack_dic=[]
for a in attack_list:
    temp_dic={}
    aa=a.split("/")[-1]
    sp=aa.split('~')
    start_timestamp=sp[0].split('.')[0]
    end_timestamp=sp[1].split('.')[0]
    temp_dic['start']=datetime_to_timestamp_US(start_timestamp)
    temp_dic['end']=datetime_to_timestamp_US(end_timestamp)
    temp_dic['file']=a
    attack_dic.append(temp_dic)
    print(temp_dic)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


original_edges_count=0
hash2msg={}
graphs=[]
gg=nx.DiGraph()
count=0
# file_list=os.listdir("./test_day_data4_10_emb100/")
for path in tqdm(attack_list):
#     print(path)
    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            count+=1
            l=line.strip()
            jdata=eval(l)
#             temp_key=jdata['srcmsg']+jdata['dstmsg']+jdata['edge_type']
#             if temp_key in train_edge_set:
#                 jdata['loss']=(jdata['loss']-train_edge_set[temp_key]) if jdata['loss']>=train_edge_set[temp_key] else 0  
#             jdata['loss']=abs(jdata['loss']-train_edge_set[temp_key])  if temp_key in train_edge_set else jdata['loss']
            edge_list.append(jdata)
            
        edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True) 
        original_edges_count+=len(edge_list)
        
        loss_list=[]
        for i in edge_list:
            loss_list.append(i['loss'])
        loss_mean=mean(loss_list)
        loss_std=std(loss_list)
        print(loss_mean)
        print(loss_std)
        thr=loss_mean+1.5*loss_std
#         thr=-99
        print("thr:",thr)
        for e in edge_list:
            if e['loss']>thr:    
#             if True:  
#                 if "'/home/admin/profile'" in e['srcmsg'] or " '/home/admin/profile'" in e['dstmsg']:
#                     print(e['srcmsg'])
#                     print(e['dstmsg'])
                tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))))
                gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))),loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
                
                hash2msg[str(hashgen(replace_path_name(e['srcmsg'])))]=replace_path_name(e['srcmsg'])
                hash2msg[str(hashgen(replace_path_name(e['dstmsg'])))]=replace_path_name(e['dstmsg'])

#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
        print(path)
        print("tempg edges:",len(tempg.edges))
        print("tempg nodes:",len(tempg.nodes))
        print("tempg weakly components:",nx.number_weakly_connected_components(tempg))
        
        print("gg edges:",len(gg.edges))
        print("gg nodes:",len(gg.nodes))
        print("gg weakly components:",nx.number_weakly_connected_components(gg))
        print(f"{original_edges_count=}")
#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
                
                
                #         graphs.append(g)

        


# In[ ]:


len(hash2msg)


# In[ ]:


# compute the best partition
import datetime
import community as community_louvain
starttime = datetime.datetime.now()
#long running
partition = community_louvain.best_partition(gg.to_undirected())
#do something other
endtime = datetime.datetime.now()
print("Finished the computation of community discovery. Execution time:{:d}".format((endtime - starttime).seconds))


# In[ ]:


communities={}
max_partition=0
for i in partition:
    if partition[i]>max_partition:
        max_partition=partition[i]
        
for i in range(max_partition+1):
    communities[i]=nx.DiGraph()
for e in gg.edges:
#     if partition[e[0]]==partition[e[1]]:
    communities[partition[e[0]]].add_edge(e[0],e[1])
    communities[partition[e[1]]].add_edge(e[0],e[1])


# In[ ]:


max_partition


# In[ ]:


def attack_edge_flag(msg):
    attack_edge_type=[
            '142.20.56.204',
        'lsass.exe',
        '142.20.61.130',
        '132.197.158.98',
        'Credentials',
    ]
    attack_edge_type=attack_node_set_h201
    flag=False
    for i in attack_edge_type:
        if i in msg:
            flag=True
    return flag


# In[ ]:


def find_time_window(edge,tw_dic):
    for t in tw_dic:
#         print(t['start'])
        if edge['time']>=t['start'] and edge['time']<=t['end']:
#             print(t['file'])
            return t['file']


# In[ ]:


max_edge_count=-99
max_index=-99
max_node_count=0

graph_index=0
for c in communities:
    file_set=set()
    if len(communities[c].edges)>max_edge_count:
        max_edge_count=len(communities[c].edges)
        max_index=graph_index
        max_node_count=len(communities[c].nodes)
    for e in communities[c].edges:    
        try:
            temp_edge=gg.edges[e]
            file_set.add(find_time_window(temp_edge,attack_dic))
        except:
            pass    
    print(f"{graph_index=}")
    print(f"file_set：")
    print(*file_set, sep = "\n")
    print(f"{file_set=}")
    graph_index+=1
    
print(f"{max_index=}")
print(f"{max_edge_count=}")
print(f"{max_node_count=}") 


# In[ ]:


file_sub_list={'graph_9_23_h201/2019-09-23 12:55:28.286~2019-09-23 13:09:50.95.txt', 'graph_9_23_h201/2019-09-23 13:10:24.95~2019-09-23 13:24:56.43.txt', 'graph_9_23_h201/2019-09-23 12:38:24.95~2019-09-23 12:54:14.286.txt', 'graph_9_23_h201/2019-09-23 11:38:40.698~2019-09-23 11:53:39.57.txt', 'graph_9_23_h201/2019-09-23 11:23:44.136~2019-09-23 11:38:30.698.txt'}


# In[ ]:


original_edges_count=0

for path in tqdm(file_sub_list):

    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            edge_list.append(line)

        original_edges_count+=len(edge_list)

        print(f"{original_edges_count=}")

        


# In[ ]:


from graphviz import Digraph


graph_index=0

total_nodes=0
total_edges=0
attck_nodes_set_visual=set()
for c in communities:
    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr['rankdir'] = 'LR'
    
    total_nodes+=len(communities[c].nodes)
    total_edges+=len(communities[c].edges)
    
    subgraph_loss_sum=0
    attack_node_count=0
    attack_edge_count=0
    for e in communities[c].edges:
        try:
            temp_edge=gg.edges[e]
            srcnode=e['srcnode']
            dstnode=e['dstnode']
        except:
            pass        

        if True:
            subgraph_loss_sum+=temp_edge['loss']
   

            if "'subject': '" in temp_edge['srcmsg']:
                src_shape='box'
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape='oval'
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape='diamond'
                
            src_shape='box'
            if attack_edge_flag(temp_edge['srcmsg']):
                src_node_color='red'
                total_nodes+=1
            else:
                src_node_color='blue'
                
            src_node_color='blue'
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                     label=str(replace_path_name(temp_edge['srcmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])), 
                     color=src_node_color,
                     shape = src_shape)


            if "'subject': '" in temp_edge['dstmsg']:
                dst_shape='box'
            elif "'file': '" in temp_edge['dstmsg']:
                dst_shape='oval'
            elif "'netflow': '" in temp_edge['dstmsg']:
                dst_shape='diamond'

                    
            if "->" in temp_edge['dstmsg']:
                dst_shape='diamond'
            else:
                dst_shape='oval'
                
                
            if attack_edge_flag(temp_edge['dstmsg']):
                dst_node_color='red'
                total_nodes+=1
            else:
                dst_node_color='blue'
            dst_node_color='blue'
            
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['dstmsg']))),
                     label=str(replace_path_name(temp_edge['dstmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])), 
                     color=dst_node_color,
                     shape = dst_shape)


    #         edgeindex=tensor_find(test_data.msg[i][16:-16],1)

    
            temp_edge_visual=replace_path_name(temp_edge['srcmsg'])+','+replace_path_name(temp_edge['dstmsg'])+','+str(temp_edge['time'])
            temp_edge_hash_val=hashgen(temp_edge_visual)
            
            if temp_edge_hash_val in edges_attack_hashset_h201:
                edge_color='red'
                attck_nodes_set_visual.add(replace_path_name(temp_edge['srcmsg']))
                attck_nodes_set_visual.add(replace_path_name(temp_edge['dstmsg']))
                attack_edge_count+=1
            else:
                edge_color='blue'
            
#             if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']):
#                 edge_color='red'
#                 attack_edge_count+=1
#             else:
#                 edge_color='blue'
                
                
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),str(hashgen(replace_path_name(temp_edge['dstmsg']))), label= temp_edge['edge_type'] , color=edge_color)#+ "  loss: "+str(temp_edge['loss']) + "  time: "+str(temp_edge['time'])

    #         dot.edge(str(srcnode), str(dstnode), label= temp_edge['edge_type']+ "  loss: "+str((temp_edge['loss'])) + "  time: "+str(temp_edge['time']) , color='red')


#     if len(communities[c].edges)<2:
#         graph_index+=1
#         continue
#     if len(communities[c].edges)>1000:
#         graph_index+=1
#         continue
#         print(f"edge num:{len(communities[c].edges)}")
        
    print("Start to render the figures···")
    
    dot.render('./graph_visual_h201/subgraph_'+str(graph_index), view=False)
    print("subgraph loss:",(subgraph_loss_sum/len(communities[c].edges)))
    print("graph_index:",graph_index)
    print("edge_count:",len(communities[c].edges))
    print("node_count:",len(communities[c].nodes))
    print(f"{attack_node_count=}")
    print(f"{attack_edge_count=}")
    graph_index+=1

print(f"avg edges:{total_edges/len(communities)}")
print(f"avg nodes:{total_nodes/len(communities)}")


# In[ ]:


attck_nodes_set_visual


# In[ ]:


len(attck_nodes_set_visual)


# In[ ]:





# In[ ]:


temp_edge_visual


# ## h402

# In[ ]:


attack_list=[
 'graph_9_23_h402/2019-09-23 13:10:24.429~2019-09-23 13:25:09.374.txt',
 'graph_9_23_h402/2019-09-23 13:10:24.429~2019-09-23 13:25:09.374.txt',
 'graph_9_23_h402/2019-09-23 13:25:20.374~2019-09-23 13:40:21.268.txt',
    'graph_9_23_h402/2019-09-23 13:40:16.268~2019-09-23 13:55:31.310.txt',
 'graph_9_23_h402/2019-09-23 13:55:12.31~2019-09-23 14:10:58.200.txt',
            ]


# In[ ]:


from datetime import datetime, timezone
attack_dic=[]
for a in attack_list:
    temp_dic={}
    aa=a.split("/")[-1]
    sp=aa.split('~')
    start_timestamp=sp[0].split('.')[0]
    end_timestamp=sp[1].split('.')[0]
    temp_dic['start']=datetime_to_timestamp_US(start_timestamp)
    temp_dic['end']=datetime_to_timestamp_US(end_timestamp)
    temp_dic['file']=a
    attack_dic.append(temp_dic)
    print(temp_dic)


# In[ ]:


replace_dic={
    '.pyc':'*.pyc',
    '.dll':'*.dll',
    '.DLL':'*.DLL',
}

def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    if '->' in path_name:
        if 'outbound' in path_name:
            msg=re.findall("->(.*?):",path_name)[0]
            return msg
        elif 'inbound' in path_name:
            msg=re.findall("#(.*?):",path_name)[0]
            return msg
    return path_name


# In[ ]:


original_edges_count=0
hash2msg={}
graphs=[]
gg=nx.DiGraph()
count=0
# file_list=os.listdir("./test_day_data4_10_emb100/")
for path in tqdm(attack_list):
#     print(path)
    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            count+=1
            l=line.strip()
            jdata=eval(l)
#             temp_key=jdata['srcmsg']+jdata['dstmsg']+jdata['edge_type']
#             if temp_key in train_edge_set:
#                 jdata['loss']=(jdata['loss']-train_edge_set[temp_key]) if jdata['loss']>=train_edge_set[temp_key] else 0  
#             jdata['loss']=abs(jdata['loss']-train_edge_set[temp_key])  if temp_key in train_edge_set else jdata['loss']
            edge_list.append(jdata)
            
        edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True) 
        original_edges_count+=len(edge_list)
        
        loss_list=[]
        for i in edge_list:
            loss_list.append(i['loss'])
        loss_mean=mean(loss_list)
        loss_std=std(loss_list)
        print(loss_mean)
        print(loss_std)
        thr=loss_mean+1.5*loss_std
#         thr=-99
        print("thr:",thr)
        for e in edge_list:
            if e['loss']>thr:    
#             if True:  
#                 if "'/home/admin/profile'" in e['srcmsg'] or " '/home/admin/profile'" in e['dstmsg']:
#                     print(e['srcmsg'])
#                     print(e['dstmsg'])
                tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))))
                gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))),loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
                
                hash2msg[str(hashgen(replace_path_name(e['srcmsg'])))]=replace_path_name(e['srcmsg'])
                hash2msg[str(hashgen(replace_path_name(e['dstmsg'])))]=replace_path_name(e['dstmsg'])
                

#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
        print(path)
        print("tempg edges:",len(tempg.edges))
        print("tempg nodes:",len(tempg.nodes))
        print("tempg weakly components:",nx.number_weakly_connected_components(tempg))
        
        print("gg edges:",len(gg.edges))
        print("gg nodes:",len(gg.nodes))
        print("gg weakly components:",nx.number_weakly_connected_components(gg))
        print(f"{original_edges_count=}")
#
#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
                
                
                #         graphs.append(g)

        


# In[ ]:


# compute the best partition
import datetime
import community as community_louvain
starttime = datetime.datetime.now()
#long running
partition = community_louvain.best_partition(gg.to_undirected())
#do something other
endtime = datetime.datetime.now()
print("Finished the computation of community discovery. Execution time:{:d}".format((endtime - starttime).seconds))


communities={}
max_partition=0
for i in partition:
    if partition[i]>max_partition:
        max_partition=partition[i]
        
for i in range(max_partition+1):
    communities[i]=nx.DiGraph()
for e in gg.edges:
#     if partition[e[0]]==partition[e[1]]:
    communities[partition[e[0]]].add_edge(e[0],e[1])
    communities[partition[e[1]]].add_edge(e[0],e[1])
    
print(f"{max_partition=}")


# In[ ]:


max_edge_count=-99
max_index=-99
max_node_count=0

graph_index=0
for c in communities:
    file_set=set()
    if len(communities[c].edges)>max_edge_count:
        max_edge_count=len(communities[c].edges)
        max_index=graph_index
        max_node_count=len(communities[c].nodes)
    for e in communities[c].edges:    
        try:
            temp_edge=gg.edges[e]
            file_set.add(find_time_window(temp_edge,attack_dic))
        except:
            pass    
    print(f"{graph_index=}")
    print(f"file_set：")
    print(*file_set, sep = "\n")
    print(f"{file_set=}")
    graph_index+=1
    
print(f"{max_index=}")
print(f"{max_edge_count=}")
print(f"{max_node_count=}") 


# In[ ]:


file_sub_list={'graph_9_23_h402/2019-09-23 13:25:20.374~2019-09-23 13:40:21.268.txt', 'graph_9_23_h402/2019-09-23 13:10:24.429~2019-09-23 13:25:09.374.txt'}




original_edges_count=0

for path in tqdm(file_sub_list):

    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            edge_list.append(line)

        original_edges_count+=len(edge_list)

        print(f"{original_edges_count=}")
#
#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
                
                
                #         graphs.append(g)

        


# In[ ]:


len(communities)


# In[ ]:





# In[ ]:


from graphviz import Digraph


graph_index=0

total_nodes=0
total_edges=0
attck_nodes_set_visual=set()
for c in communities:

    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr['rankdir'] = 'LR'
    # dot.node(name='a', label='wo', color='purple')
    # dot.node(name='b', label='niu', color='purple')
    # dot.node(name='c', label='che', color='purple')
    
    total_nodes+=len(communities[c].nodes)
    total_edges+=len(communities[c].edges)
    
    subgraph_loss_sum=0
    attack_node_count=0
    attack_edge_count=0
    for e in communities[c].edges:
        try:
            temp_edge=gg.edges[e]
            srcnode=e['srcnode']
            dstnode=e['dstnode']
        except:
            pass        

        if True:
            subgraph_loss_sum+=temp_edge['loss']

    #     g.add_edge(srcnode,dstnode,srcmsg=node2msg[indexid2nodeid[srcnode]],dstmsg=node2msg[indexid2nodeid[dstnode]],loss=df['loss'][i])


    #         dot.node( name=str(srcnode),label=str(node_index_id2msg_mal[srcnode]), color='purple',shape = 'box')
            if "'subject': '" in temp_edge['srcmsg']:
                src_shape='box'
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape='oval'
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape='diamond'
                
            src_shape='box'
            if attack_edge_flag(temp_edge['srcmsg']):
                src_node_color='red'
                total_nodes+=1
            else:
                src_node_color='blue'
                
            src_node_color='blue'
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                     label=str(replace_path_name(temp_edge['srcmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])), 
                     color=src_node_color,
                     shape = src_shape)


            if "'subject': '" in temp_edge['dstmsg']:
                dst_shape='box'
            elif "'file': '" in temp_edge['dstmsg']:
                dst_shape='oval'
            elif "'netflow': '" in temp_edge['dstmsg']:
                dst_shape='diamond'

                    
            if "->" in temp_edge['dstmsg']:
                dst_shape='diamond'
            else:
                dst_shape='oval'
                
                
            if attack_edge_flag(temp_edge['dstmsg']):
                dst_node_color='red'
                total_nodes+=1
            else:
                dst_node_color='blue'
            dst_node_color='blue'
            
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['dstmsg']))),
                     label=str(replace_path_name(temp_edge['dstmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])), 
                     color=dst_node_color,
                     shape = dst_shape)


    #         edgeindex=tensor_find(test_data.msg[i][16:-16],1)

    
            temp_edge_visual=replace_path_name(temp_edge['srcmsg'])+','+replace_path_name(temp_edge['dstmsg'])+','+str(temp_edge['time'])
            temp_edge_hash_val=hashgen(temp_edge_visual)
            
            if temp_edge_hash_val in edges_attack_hashset_h402:
                edge_color='red'
                attck_nodes_set_visual.add(replace_path_name(temp_edge['srcmsg']))
                attck_nodes_set_visual.add(replace_path_name(temp_edge['dstmsg']))
                attack_edge_count+=1
            else:
                edge_color='blue'
            
#             if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']):
#                 edge_color='red'
#                 attack_edge_count+=1
#             else:
#                 edge_color='blue'
                
                
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),str(hashgen(replace_path_name(temp_edge['dstmsg']))), label= temp_edge['edge_type'] , color=edge_color)#+ "  loss: "+str(temp_edge['loss']) + "  time: "+str(temp_edge['time'])

    #         dot.edge(str(srcnode), str(dstnode), label= temp_edge['edge_type']+ "  loss: "+str((temp_edge['loss'])) + "  time: "+str(temp_edge['time']) , color='red')


    if len(communities[c].edges)<2:
        graph_index+=1
        continue
    if len(communities[c].edges)>1000:
        graph_index+=1
        continue
        print(f"edge num:{len(communities[c].edges)}，skip rendering")
        
    print("Start to reder the figures···")
    
    dot.render('./graph_visual_h402/subgraph_'+str(graph_index), view=False)
    print("subgraph loss:",(subgraph_loss_sum/len(communities[c].edges)))
    print("graph_index:",graph_index)
    print("edge_count:",len(communities[c].edges))
    print("node_count:",len(communities[c].nodes))
    print(f"{attack_node_count=}")
    print(f"{attack_edge_count=}")
    graph_index+=1

print(f"avg edges:{total_edges/len(communities)}")
print(f"avg nodes:{total_nodes/len(communities)}")


# In[ ]:





# In[ ]:


attck_nodes_set_visual


# In[ ]:





# In[ ]:





# ## h660

# In[ ]:


attack_list=[
    
    'graph_9_23_h660/2019-09-23 09:54:08.697~2019-09-23 10:09:11.523.txt',
'graph_9_23_h660/2019-09-23 13:27:28.512~2019-09-23 13:42:30.682.txt',
 'graph_9_23_h660/2019-09-23 13:42:24.682~2019-09-23 13:57:57.566.txt',
 'graph_9_23_h660/2019-09-23 13:57:20.566~2019-09-23 14:12:59.139.txt',
            ]


# In[ ]:


from datetime import datetime, timezone
attack_dic=[]
for a in attack_list:
    temp_dic={}
    aa=a.split("/")[-1]
    sp=aa.split('~')
    start_timestamp=sp[0].split('.')[0]
    end_timestamp=sp[1].split('.')[0]
    temp_dic['start']=datetime_to_timestamp_US(start_timestamp)
    temp_dic['end']=datetime_to_timestamp_US(end_timestamp)
    temp_dic['file']=a
    attack_dic.append(temp_dic)
    print(temp_dic)


# In[ ]:


replace_dic={
    '.pyc':'*.pyc',
#     '.dll':'*.dll',
#     '.DLL':'*.DLL',
}

def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    if '->' in path_name:
        if 'outbound' in path_name:
            msg=re.findall("->(.*?):",path_name)[0]
            return msg
        elif 'inbound' in path_name:
            msg=re.findall("#(.*?):",path_name)[0]
            return msg
    return path_name


# In[ ]:


original_edges_count=0
hash2msg={}
graphs=[]
gg=nx.DiGraph()
count=0
# file_list=os.listdir("./test_day_data4_10_emb100/")
for path in tqdm(attack_list):
#     print(path)
    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            count+=1
            l=line.strip()
            jdata=eval(l)
#             temp_key=jdata['srcmsg']+jdata['dstmsg']+jdata['edge_type']
#             if temp_key in train_edge_set:
#                 jdata['loss']=(jdata['loss']-train_edge_set[temp_key]) if jdata['loss']>=train_edge_set[temp_key] else 0  
#             jdata['loss']=abs(jdata['loss']-train_edge_set[temp_key])  if temp_key in train_edge_set else jdata['loss']
            edge_list.append(jdata)
            
        edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True) 
        original_edges_count+=len(edge_list)
        
        loss_list=[]
        for i in edge_list:
            loss_list.append(i['loss'])
        loss_mean=mean(loss_list)
        loss_std=std(loss_list)
        print(loss_mean)
        print(loss_std)
        thr=loss_mean+1.5*loss_std
#         thr=-99
        print("thr:",thr)
        for e in edge_list:
            if e['loss']>thr:    
#             if True:  
#                 if "'/home/admin/profile'" in e['srcmsg'] or " '/home/admin/profile'" in e['dstmsg']:
#                     print(e['srcmsg'])
#                     print(e['dstmsg'])
                tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))))
                gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))),loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
                
                hash2msg[str(hashgen(replace_path_name(e['srcmsg'])))]=replace_path_name(e['srcmsg'])
                hash2msg[str(hashgen(replace_path_name(e['dstmsg'])))]=replace_path_name(e['dstmsg'])
                

#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
        print(path)
        print("tempg edges:",len(tempg.edges))
        print("tempg nodes:",len(tempg.nodes))
        print("tempg weakly components:",nx.number_weakly_connected_components(tempg))
        
        print("gg edges:",len(gg.edges))
        print("gg nodes:",len(gg.nodes))
        print("gg weakly components:",nx.number_weakly_connected_components(gg))
        print(f"{original_edges_count=}")
#
#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
                
                
                #         graphs.append(g)

        


# In[ ]:


# compute the best partition
import datetime
import community as community_louvain
starttime = datetime.datetime.now()
#long running
partition = community_louvain.best_partition(gg.to_undirected())
#do something other
endtime = datetime.datetime.now()
print("Finished the computation of community discovery. Execution time:{:d}".format((endtime - starttime).seconds))


communities={}
max_partition=0
for i in partition:
    if partition[i]>max_partition:
        max_partition=partition[i]
        
for i in range(max_partition+1):
    communities[i]=nx.DiGraph()
for e in gg.edges:
#     if partition[e[0]]==partition[e[1]]:
    communities[partition[e[0]]].add_edge(e[0],e[1])
    communities[partition[e[1]]].add_edge(e[0],e[1])
    
print(f"{max_partition=}")


# In[ ]:


max_edge_count=-99
max_index=-99
max_node_count=0

graph_index=0
for c in communities:
    file_set=set()
    if len(communities[c].edges)>max_edge_count:
        max_edge_count=len(communities[c].edges)
        max_index=graph_index
        max_node_count=len(communities[c].nodes)
    for e in communities[c].edges:    
        try:
            temp_edge=gg.edges[e]
            file_set.add(find_time_window(temp_edge,attack_dic))
        except:
            pass    
    print(f"{graph_index=}")
    print(f"file_set：")
    print(*file_set, sep = "\n")
    print(f"{file_set=}")
    graph_index+=1
    
print(f"{max_index=}")
print(f"{max_edge_count=}")
print(f"{max_node_count=}") 


# In[ ]:


file_sub_list={'graph_9_23_h660/2019-09-23 13:27:28.512~2019-09-23 13:42:30.682.txt', 'graph_9_23_h660/2019-09-23 13:42:24.682~2019-09-23 13:57:57.566.txt', 'graph_9_23_h660/2019-09-23 13:57:20.566~2019-09-23 14:12:59.139.txt'}



original_edges_count=0

for path in tqdm(file_sub_list):

    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            edge_list.append(line)

        original_edges_count+=len(edge_list)

        print(f"{original_edges_count=}")
#
#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
                
                
                #         graphs.append(g)

        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from graphviz import Digraph


graph_index=0

total_nodes=0
total_edges=0
attck_nodes_set_visual=set()
for c in communities:
    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr['rankdir'] = 'LR'
    # dot.node(name='a', label='wo', color='purple')
    # dot.node(name='b', label='niu', color='purple')
    # dot.node(name='c', label='che', color='purple')
    
    total_nodes+=len(communities[c].nodes)
    total_edges+=len(communities[c].edges)
    
    subgraph_loss_sum=0
    attack_node_count=0
    attack_edge_count=0
    for e in communities[c].edges:
        try:
            temp_edge=gg.edges[e]
            srcnode=e['srcnode']
            dstnode=e['dstnode']
        except:
            pass        

        if True:
            subgraph_loss_sum+=temp_edge['loss']

    #     g.add_edge(srcnode,dstnode,srcmsg=node2msg[indexid2nodeid[srcnode]],dstmsg=node2msg[indexid2nodeid[dstnode]],loss=df['loss'][i])


    #         dot.node( name=str(srcnode),label=str(node_index_id2msg_mal[srcnode]), color='purple',shape = 'box')
            if "'subject': '" in temp_edge['srcmsg']:
                src_shape='box'
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape='oval'
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape='diamond'
                
            src_shape='box'
            if attack_edge_flag(temp_edge['srcmsg']):
                src_node_color='red'
                total_nodes+=1
            else:
                src_node_color='blue'
                
            src_node_color='blue'
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                     label=str(replace_path_name(temp_edge['srcmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])), 
                     color=src_node_color,
                     shape = src_shape)


            if "'subject': '" in temp_edge['dstmsg']:
                dst_shape='box'
            elif "'file': '" in temp_edge['dstmsg']:
                dst_shape='oval'
            elif "'netflow': '" in temp_edge['dstmsg']:
                dst_shape='diamond'

                    
            if "->" in temp_edge['dstmsg']:
                dst_shape='diamond'
            else:
                dst_shape='oval'
                
                
            if attack_edge_flag(temp_edge['dstmsg']):
                dst_node_color='red'
                total_nodes+=1
            else:
                dst_node_color='blue'
            dst_node_color='blue'
            
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['dstmsg']))),
                     label=str(replace_path_name(temp_edge['dstmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])), 
                     color=dst_node_color,
                     shape = dst_shape)


    #         edgeindex=tensor_find(test_data.msg[i][16:-16],1)

    
            temp_edge_visual=replace_path_name(temp_edge['srcmsg'])+','+replace_path_name(temp_edge['dstmsg'])+','+str(temp_edge['time'])
            temp_edge_hash_val=hashgen(temp_edge_visual)
            
            if temp_edge_hash_val in edges_attack_hashset_h660:
                edge_color='red'
                attck_nodes_set_visual.add(replace_path_name(temp_edge['srcmsg']))
                attck_nodes_set_visual.add(replace_path_name(temp_edge['dstmsg']))
                attack_edge_count+=1
            else:
                edge_color='blue'
            
#             if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']):
#                 edge_color='red'
#                 attack_edge_count+=1
#             else:
#                 edge_color='blue'
                
                
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),str(hashgen(replace_path_name(temp_edge['dstmsg']))), label= temp_edge['edge_type'] , color=edge_color)#+ "  loss: "+str(temp_edge['loss']) + "  time: "+str(temp_edge['time'])

    #         dot.edge(str(srcnode), str(dstnode), label= temp_edge['edge_type']+ "  loss: "+str((temp_edge['loss'])) + "  time: "+str(temp_edge['time']) , color='red')


#     if len(communities[c].edges)<2:
#         continue
#     if len(communities[c].edges)>1000:
#         continue
#         print(f"edge num:{len(communities[c].edges)}，skip rendering")
        
    print("Start to render the figures···")
    
    dot.render('./graph_visual_h660/subgraph_'+str(graph_index), view=False)
    print("subgraph loss:",(subgraph_loss_sum/len(communities[c].edges)))
    print("graph_index:",graph_index)
    print("edge_count:",len(communities[c].edges))
    print("node_count:",len(communities[c].nodes))
    print(f"{attack_node_count=}")
    print(f"{attack_edge_count=}")
    graph_index+=1

print(f"avg edges:{total_edges/len(communities)}")
print(f"avg nodes:{total_nodes/len(communities)}")


# In[ ]:


attck_nodes_set_visual


# In[ ]:





# In[ ]:





# ## h501

# In[ ]:


attack_list=[
 'graph_9_24_h501/2019-09-24 10:15:28.241~2019-09-24 10:30:00.201.txt',
 'graph_9_24_h501/2019-09-24 10:30:24.201~2019-09-24 10:45:02.7.txt',
 'graph_9_24_h501/2019-09-24 10:45:20.7~2019-09-24 11:00:31.385.txt',
 'graph_9_24_h501/2019-09-24 11:00:16.385~2019-09-24 11:16:09.755.txt',
 'graph_9_24_h501/2019-09-24 11:15:12.755~2019-09-24 11:31:14.287.txt',
 'graph_9_24_h501/2019-09-24 11:32:16.287~2019-09-24 11:46:31.541.txt',
    
 'graph_9_24_h501/2019-09-24 13:04:00.804~2019-09-24 13:17:29.451.txt',
 'graph_9_24_h501/2019-09-24 13:18:56.451~2019-09-24 13:32:46.454.txt',
 'graph_9_24_h501/2019-09-24 13:33:52.454~2019-09-24 13:48:02.493.txt',
            ]


# In[ ]:


from datetime import datetime, timezone
attack_dic=[]
for a in attack_list:
    temp_dic={}
    aa=a.split("/")[-1]
    sp=aa.split('~')
    start_timestamp=sp[0].split('.')[0]
    end_timestamp=sp[1].split('.')[0]
    temp_dic['start']=datetime_to_timestamp_US(start_timestamp)
    temp_dic['end']=datetime_to_timestamp_US(end_timestamp)
    temp_dic['file']=a
    attack_dic.append(temp_dic)
    print(temp_dic)


# In[ ]:


replace_dic={
    '.pyc':'*.pyc',
    '.dll':'*.dll',
    '.DLL':'*.DLL',
}

def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    if '->' in path_name:
        if 'outbound' in path_name:
            msg=re.findall("->(.*?):",path_name)[0]
            return msg
        elif 'inbound' in path_name:
            msg=re.findall("#(.*?):",path_name)[0]
            return msg
    return path_name


# In[ ]:


original_edges_count=0
hash2msg={}
graphs=[]
gg=nx.DiGraph()
count=0
# file_list=os.listdir("./test_day_data4_10_emb100/")
for path in tqdm(attack_list):
#     print(path)
    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            count+=1
            l=line.strip()
            jdata=eval(l)
#             temp_key=jdata['srcmsg']+jdata['dstmsg']+jdata['edge_type']
#             if temp_key in train_edge_set:
#                 jdata['loss']=(jdata['loss']-train_edge_set[temp_key]) if jdata['loss']>=train_edge_set[temp_key] else 0  
#             jdata['loss']=abs(jdata['loss']-train_edge_set[temp_key])  if temp_key in train_edge_set else jdata['loss']
            edge_list.append(jdata)
            
        edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True) 
        original_edges_count+=len(edge_list)
        
        loss_list=[]
        for i in edge_list:
            loss_list.append(i['loss'])
        loss_mean=mean(loss_list)
        loss_std=std(loss_list)
        print(loss_mean)
        print(loss_std)
        thr=loss_mean+2.5*loss_std
#         thr=-99
        print("thr:",thr)
        for e in edge_list:
            if e['loss']>thr:    
#             if True:  
#                 if "'/home/admin/profile'" in e['srcmsg'] or " '/home/admin/profile'" in e['dstmsg']:
#                     print(e['srcmsg'])
#                     print(e['dstmsg'])
                tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))))
                gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))),loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
                
                hash2msg[str(hashgen(replace_path_name(e['srcmsg'])))]=replace_path_name(e['srcmsg'])
                hash2msg[str(hashgen(replace_path_name(e['dstmsg'])))]=replace_path_name(e['dstmsg'])
                

#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
        print(path)
        print("tempg edges:",len(tempg.edges))
        print("tempg nodes:",len(tempg.nodes))
        print("tempg weakly components:",nx.number_weakly_connected_components(tempg))
        
        print("gg edges:",len(gg.edges))
        print("gg nodes:",len(gg.nodes))
        print("gg weakly components:",nx.number_weakly_connected_components(gg))
        print(f"{original_edges_count=}")
#
#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
                
                
                #         graphs.append(g)

        


# In[ ]:


# compute the best partition
import datetime
import community as community_louvain
starttime = datetime.datetime.now()
#long running
partition = community_louvain.best_partition(gg.to_undirected())
#do something other
endtime = datetime.datetime.now()
print("Finished the computation of community discovery. Execution time:{:d}".format((endtime - starttime).seconds))


communities={}
max_partition=0
for i in partition:
    if partition[i]>max_partition:
        max_partition=partition[i]
        
for i in range(max_partition+1):
    communities[i]=nx.DiGraph()
for e in gg.edges:
#     if partition[e[0]]==partition[e[1]]:
    communities[partition[e[0]]].add_edge(e[0],e[1])
    communities[partition[e[1]]].add_edge(e[0],e[1])
    
print(f"{max_partition=}")


# In[ ]:


max_edge_count=-99
max_index=-99
max_node_count=0

graph_index=0
for c in communities:
    file_set=set()
    if len(communities[c].edges)>max_edge_count:
        max_edge_count=len(communities[c].edges)
        max_index=graph_index
        max_node_count=len(communities[c].nodes)
    for e in communities[c].edges:    
        try:
            temp_edge=gg.edges[e]
            file_set.add(find_time_window(temp_edge,attack_dic))
        except:
            pass    
    print(f"{graph_index=}")
    print(f"file_set：")
    print(*file_set, sep = "\n")
    print(f"{file_set=}")
    graph_index+=1
    
print(f"{max_index=}")
print(f"{max_edge_count=}")
print(f"{max_node_count=}") 


# In[ ]:


file_sub_list={'graph_9_24_h501/2019-09-24 10:45:20.7~2019-09-24 11:00:31.385.txt', 'graph_9_24_h501/2019-09-24 11:00:16.385~2019-09-24 11:16:09.755.txt', 'graph_9_24_h501/2019-09-24 10:15:28.241~2019-09-24 10:30:00.201.txt', 'graph_9_24_h501/2019-09-24 11:15:12.755~2019-09-24 11:31:14.287.txt', 'graph_9_24_h501/2019-09-24 11:32:16.287~2019-09-24 11:46:31.541.txt', 'graph_9_24_h501/2019-09-24 10:30:24.201~2019-09-24 10:45:02.7.txt','graph_9_24_h501/2019-09-24 10:15:28.241~2019-09-24 10:30:00.201.txt', 'graph_9_24_h501/2019-09-24 13:04:00.804~2019-09-24 13:17:29.451.txt', 'graph_9_24_h501/2019-09-24 11:15:12.755~2019-09-24 11:31:14.287.txt', 'graph_9_24_h501/2019-09-24 11:32:16.287~2019-09-24 11:46:31.541.txt', 'graph_9_24_h501/2019-09-24 13:33:52.454~2019-09-24 13:48:02.493.txt', 'graph_9_24_h501/2019-09-24 10:30:24.201~2019-09-24 10:45:02.7.txt', 'graph_9_24_h501/2019-09-24 13:18:56.451~2019-09-24 13:32:46.454.txt'}

original_edges_count=0

for path in tqdm(file_sub_list):

    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            edge_list.append(line)

        original_edges_count+=len(edge_list)

        print(f"{original_edges_count=}")
#
#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
                
                
                #         graphs.append(g)

        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from graphviz import Digraph


graph_index=0

total_nodes=0
total_edges=0
attck_nodes_set_visual=set()
for c in communities:
    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr['rankdir'] = 'LR'
    # dot.node(name='a', label='wo', color='purple')
    # dot.node(name='b', label='niu', color='purple')
    # dot.node(name='c', label='che', color='purple')
    
    total_nodes+=len(communities[c].nodes)
    total_edges+=len(communities[c].edges)
    
    subgraph_loss_sum=0
    attack_node_count=0
    attack_edge_count=0
    for e in communities[c].edges:
        try:
            temp_edge=gg.edges[e]
            srcnode=e['srcnode']
            dstnode=e['dstnode']
        except:
            pass        

        if True:
            subgraph_loss_sum+=temp_edge['loss']

    #     g.add_edge(srcnode,dstnode,srcmsg=node2msg[indexid2nodeid[srcnode]],dstmsg=node2msg[indexid2nodeid[dstnode]],loss=df['loss'][i])


    #         dot.node( name=str(srcnode),label=str(node_index_id2msg_mal[srcnode]), color='purple',shape = 'box')
            if "'subject': '" in temp_edge['srcmsg']:
                src_shape='box'
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape='oval'
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape='diamond'
                
            src_shape='box'
            if attack_edge_flag(temp_edge['srcmsg']):
                src_node_color='red'
                total_nodes+=1
            else:
                src_node_color='blue'
                
            src_node_color='blue'
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                     label=str(replace_path_name(temp_edge['srcmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])), 
                     color=src_node_color,
                     shape = src_shape)


            if "'subject': '" in temp_edge['dstmsg']:
                dst_shape='box'
            elif "'file': '" in temp_edge['dstmsg']:
                dst_shape='oval'
            elif "'netflow': '" in temp_edge['dstmsg']:
                dst_shape='diamond'

                    
            if "->" in temp_edge['dstmsg']:
                dst_shape='diamond'
            else:
                dst_shape='oval'
                
                
            if attack_edge_flag(temp_edge['dstmsg']):
                dst_node_color='red'
                total_nodes+=1
            else:
                dst_node_color='blue'
            dst_node_color='blue'
            
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['dstmsg']))),
                     label=str(replace_path_name(temp_edge['dstmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])), 
                     color=dst_node_color,
                     shape = dst_shape)


    #         edgeindex=tensor_find(test_data.msg[i][16:-16],1)
    
            temp_edge_visual=replace_path_name(temp_edge['srcmsg'])+','+replace_path_name(temp_edge['dstmsg'])+','+str(temp_edge['time'])
            temp_edge_hash_val=hashgen(temp_edge_visual)
            
            if temp_edge_hash_val in edges_attack_hashset_h501:
                edge_color='red'
                attck_nodes_set_visual.add(replace_path_name(temp_edge['srcmsg']))
                attck_nodes_set_visual.add(replace_path_name(temp_edge['dstmsg']))
                attack_edge_count+=1
            else:
                edge_color='blue'
            
#             if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']):
#                 edge_color='red'
#                 attack_edge_count+=1
#             else:
#                 edge_color='blue'
                
                
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),str(hashgen(replace_path_name(temp_edge['dstmsg']))), label= temp_edge['edge_type'] , color=edge_color)#+ "  loss: "+str(temp_edge['loss']) + "  time: "+str(temp_edge['time'])

    #         dot.edge(str(srcnode), str(dstnode), label= temp_edge['edge_type']+ "  loss: "+str((temp_edge['loss'])) + "  time: "+str(temp_edge['time']) , color='red')


#     if len(communities[c].edges)<2:
#         continue
#     if len(communities[c].edges)>1000:
#         continue
#         print(f"edge num:{len(communities[c].edges)}，skip rendering")
        
    print("Start to render the figures···")
    
    dot.render('./graph_visual_h501/subgraph_'+str(graph_index), view=False)
    print("subgraph loss:",(subgraph_loss_sum/len(communities[c].edges)))
    print("graph_index:",graph_index)
    print("edge_count:",len(communities[c].edges))
    print("node_count:",len(communities[c].nodes))
    print(f"{attack_node_count=}")
    print(f"{attack_edge_count=}")
    graph_index+=1

print(f"avg edges:{total_edges/len(communities)}")
print(f"avg nodes:{total_nodes/len(communities)}")


# In[ ]:


attck_nodes_set_visual


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## h051

# In[ ]:


attack_list=[
    'graph_9_25_h051/2019-09-25 09:11:28.846~2019-09-25 09:25:34.393.txt',
    'graph_9_25_h051/2019-09-25 09:26:24.393~2019-09-25 09:40:36.393.txt',
'graph_9_25_h051/2019-09-25 10:26:08.397~2019-09-25 10:41:40.247.txt',
 'graph_9_25_h051/2019-09-25 10:41:04.247~2019-09-25 10:56:56.92.txt',
 'graph_9_25_h051/2019-09-25 10:56:00.92~2019-09-25 11:12:03.608.txt',
    'graph_9_25_h051/2019-09-25 14:16:32.762~2019-09-25 14:32:02.764.txt',
            ]


# In[ ]:


from datetime import datetime, timezone
attack_dic=[]
for a in attack_list:
    temp_dic={}
    aa=a.split("/")[-1]
    sp=aa.split('~')
    start_timestamp=sp[0].split('.')[0]
    end_timestamp=sp[1].split('.')[0]
    temp_dic['start']=datetime_to_timestamp_US(start_timestamp)
    temp_dic['end']=datetime_to_timestamp_US(end_timestamp)
    temp_dic['file']=a
    attack_dic.append(temp_dic)
    print(temp_dic)


# In[ ]:


replace_dic={
    '.pyc':'*.pyc',
    '.dll':'*.dll',
    '.DLL':'*.DLL',
}

def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    if '->' in path_name:
        if 'outbound' in path_name:
            msg=re.findall("->(.*?):",path_name)[0]
            return msg
        elif 'inbound' in path_name:
            msg=re.findall("#(.*?):",path_name)[0]
            return msg
    return path_name


# In[ ]:


original_edges_count=0
hash2msg={}
graphs=[]
gg=nx.DiGraph()
count=0
# file_list=os.listdir("./test_day_data4_10_emb100/")
for path in tqdm(attack_list):
#     print(path)
    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            count+=1
            l=line.strip()
            jdata=eval(l)
#             temp_key=jdata['srcmsg']+jdata['dstmsg']+jdata['edge_type']
#             if temp_key in train_edge_set:
#                 jdata['loss']=(jdata['loss']-train_edge_set[temp_key]) if jdata['loss']>=train_edge_set[temp_key] else 0  
#             jdata['loss']=abs(jdata['loss']-train_edge_set[temp_key])  if temp_key in train_edge_set else jdata['loss']
            edge_list.append(jdata)
            
        edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True) 
        original_edges_count+=len(edge_list)
        
        loss_list=[]
        for i in edge_list:
            loss_list.append(i['loss'])
        loss_mean=mean(loss_list)
        loss_std=std(loss_list)
        print(loss_mean)
        print(loss_std)
        thr=loss_mean+2.5*loss_std
#         thr=-99
        print("thr:",thr)
        for e in edge_list:
            if e['loss']>thr:    
#             if True:  
#                 if "'/home/admin/profile'" in e['srcmsg'] or " '/home/admin/profile'" in e['dstmsg']:
#                     print(e['srcmsg'])
#                     print(e['dstmsg'])
                tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))))
                gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))),loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
                
                hash2msg[str(hashgen(replace_path_name(e['srcmsg'])))]=replace_path_name(e['srcmsg'])
                hash2msg[str(hashgen(replace_path_name(e['dstmsg'])))]=replace_path_name(e['dstmsg'])
                

#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
        print(path)
        print("tempg edges:",len(tempg.edges))
        print("tempg nodes:",len(tempg.nodes))
        print("tempg weakly components:",nx.number_weakly_connected_components(tempg))
        
        print("gg edges:",len(gg.edges))
        print("gg nodes:",len(gg.nodes))
        print("gg weakly components:",nx.number_weakly_connected_components(gg))
        print(f"{original_edges_count=}")
#
#                 gg.add_edge(e['srcnode'],e['dstnode'],loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
#         print(path," line_count:",line_count,"  nodes count:",len(node_set))
                
                
                #         graphs.append(g)

        


# In[ ]:


# compute the best partition
import datetime
import community as community_louvain
starttime = datetime.datetime.now()
#long running
partition = community_louvain.best_partition(gg.to_undirected())
#do something other
endtime = datetime.datetime.now()
print("Finished the computation of community discovery. Execution time:{:d}".format((endtime - starttime).seconds))


communities={}
max_partition=0
for i in partition:
    if partition[i]>max_partition:
        max_partition=partition[i]
        
for i in range(max_partition+1):
    communities[i]=nx.DiGraph()
for e in gg.edges:
#     if partition[e[0]]==partition[e[1]]:
    communities[partition[e[0]]].add_edge(e[0],e[1])
    communities[partition[e[1]]].add_edge(e[0],e[1])
    
print(f"{max_partition=}")


# In[ ]:


from graphviz import Digraph


graph_index=0

total_nodes=0
total_edges=0
attck_nodes_set_visual=set()
for c in communities:
    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr['rankdir'] = 'LR'
    # dot.node(name='a', label='wo', color='purple')
    # dot.node(name='b', label='niu', color='purple')
    # dot.node(name='c', label='che', color='purple')
    
    total_nodes+=len(communities[c].nodes)
    total_edges+=len(communities[c].edges)
    
    subgraph_loss_sum=0
    attack_node_count=0
    attack_edge_count=0
    for e in communities[c].edges:
        try:
            temp_edge=gg.edges[e]
            srcnode=e['srcnode']
            dstnode=e['dstnode']
        except:
            pass        

        if True:
            subgraph_loss_sum+=temp_edge['loss']

    #     g.add_edge(srcnode,dstnode,srcmsg=node2msg[indexid2nodeid[srcnode]],dstmsg=node2msg[indexid2nodeid[dstnode]],loss=df['loss'][i])


    #         dot.node( name=str(srcnode),label=str(node_index_id2msg_mal[srcnode]), color='purple',shape = 'box')
            if "'subject': '" in temp_edge['srcmsg']:
                src_shape='box'
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape='oval'
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape='diamond'
                
            src_shape='box'
            if attack_edge_flag(temp_edge['srcmsg']):
                src_node_color='red'
                total_nodes+=1
            else:
                src_node_color='blue'
                
            src_node_color='blue'
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                     label=str(replace_path_name(temp_edge['srcmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])), 
                     color=src_node_color,
                     shape = src_shape)


            if "'subject': '" in temp_edge['dstmsg']:
                dst_shape='box'
            elif "'file': '" in temp_edge['dstmsg']:
                dst_shape='oval'
            elif "'netflow': '" in temp_edge['dstmsg']:
                dst_shape='diamond'

                    
            if "->" in temp_edge['dstmsg']:
                dst_shape='diamond'
            else:
                dst_shape='oval'
                
                
            if attack_edge_flag(temp_edge['dstmsg']):
                dst_node_color='red'
                total_nodes+=1
            else:
                dst_node_color='blue'
            dst_node_color='blue'
            
            
            dot.node( name=str(hashgen(replace_path_name(temp_edge['dstmsg']))),
                     label=str(replace_path_name(temp_edge['dstmsg'])+'\t partition:'+str(partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])), 
                     color=dst_node_color,
                     shape = dst_shape)


    #         edgeindex=tensor_find(test_data.msg[i][16:-16],1)
    
            temp_edge_visual=replace_path_name(temp_edge['srcmsg'])+','+replace_path_name(temp_edge['dstmsg'])+','+str(temp_edge['time'])
            temp_edge_hash_val=hashgen(temp_edge_visual)
            
            if temp_edge_hash_val in edges_attack_hashset_h051:
                edge_color='red'
                attck_nodes_set_visual.add(replace_path_name(temp_edge['srcmsg']))
                attck_nodes_set_visual.add(replace_path_name(temp_edge['dstmsg']))
                attack_edge_count+=1
            else:
                edge_color='blue'
            
#             if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']):
#                 edge_color='red'
#                 attack_edge_count+=1
#             else:
#                 edge_color='blue'
                
                
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),str(hashgen(replace_path_name(temp_edge['dstmsg']))), label= temp_edge['edge_type'] , color=edge_color)#+ "  loss: "+str(temp_edge['loss']) + "  time: "+str(temp_edge['time'])

    #         dot.edge(str(srcnode), str(dstnode), label= temp_edge['edge_type']+ "  loss: "+str((temp_edge['loss'])) + "  time: "+str(temp_edge['time']) , color='red')


#     if len(communities[c].edges)<2:
#         continue
#     if len(communities[c].edges)>1000:
#         continue
#         print(f"edge num:{len(communities[c].edges)}，skip rendering")
        
    print("Start to render the figures···")
    
    dot.render('./graph_visual_h051/subgraph_'+str(graph_index), view=False)
    print("subgraph loss:",(subgraph_loss_sum/len(communities[c].edges)))
    print("graph_index:",graph_index)
    print("edge_count:",len(communities[c].edges))
    print("node_count:",len(communities[c].nodes))
    print(f"{attack_node_count=}")
    print(f"{attack_edge_count=}")
    graph_index+=1

print(f"avg edges:{total_edges/len(communities)}")
print(f"avg nodes:{total_nodes/len(communities)}")

