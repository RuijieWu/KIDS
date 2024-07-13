#!/usr/bin/env python
# coding: utf-8

# In[32]:


# encoding=utf-8
import os.path as osp
import os
import copy
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import TemporalData
from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import ICEWS18
from torch_geometric.nn import TGNMemory, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage, MeanAggregator,
                                           LastAggregator)
from torch_geometric import *
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# msg structure:    [src_node_feature,edge_attr,dst_node_feature]

# compute the best partition 
import datetime
# import community as community_louvain

import xxhash

# Find the edge index which the edge vector is corresponding to
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

def cal_pos_edges_loss_autoencoder(decoded,msg):
    loss=[] 
    for i in range(len(decoded)):
        loss.append(criterion(decoded[i].reshape(1,-1),msg[i].reshape(-1)))
    return torch.tensor(loss)


# In[2]:


get_ipython().run_line_magic('autosave', '120')


# In[3]:


from datetime import datetime, timezone
import time
import pytz
from time import mktime
from datetime import datetime
import time
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





# In[4]:


import psycopg2

from psycopg2 import extras as ex
connect = psycopg2.connect(database = 'tc_e5_cadets_dataset_db',
                           host = '/var/run/postgresql/',
                           user = 'postgres',
                           password = 'postgres',
                           port = '5432'
                          )


cur = connect.cursor()


# In[6]:


graph_5_8=torch.load("./train_graph/graph_5_8.TemporalData.simple").to(device=device)
graph_5_9=torch.load("./train_graph/graph_5_9.TemporalData.simple").to(device=device)
graph_5_11=torch.load("./train_graph/graph_5_11.TemporalData.simple").to(device=device)


train_data=graph_5_8


# In[ ]:





# In[7]:


# Constructing the map for nodeid to msg
sql="select * from node2id ORDER BY index_id;"
cur.execute(sql)
rows = cur.fetchall()

nodeid2msg={}  # nodeid => msg and node hash => nodeid
for i in rows:
    nodeid2msg[i[0]]=i[-1]
    nodeid2msg[i[-1]]={i[1]:i[2]}  


# In[8]:


rel2id={1: 'EVENT_CLOSE',
 'EVENT_CLOSE': 1,
 2: 'EVENT_OPEN',
 'EVENT_OPEN': 2,
 3: 'EVENT_READ',
 'EVENT_READ': 3,
 4: 'EVENT_WRITE',
 'EVENT_WRITE': 4,
 5: 'EVENT_EXECUTE',
 'EVENT_EXECUTE': 5,
 6: 'EVENT_RECVFROM',
 'EVENT_RECVFROM': 6,
 7: 'EVENT_RECVMSG',
 'EVENT_RECVMSG': 7,
 8: 'EVENT_SENDMSG',
 'EVENT_SENDMSG': 8,
 9: 'EVENT_SENDTO',
 'EVENT_SENDTO': 9}


# In[ ]:





# In[9]:


# train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
# max_node_num = max(torch.cat([data.dst,data.src]))+1
# max_node_num = data.num_nodes+1
max_node_num = 262626  # +1
# min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
min_dst_idx, max_dst_idx = 0, max_node_num
neighbor_loader = LastNeighborLoader(max_node_num, size=20, device=device)


# In[ ]:





# In[10]:


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
            
            Linear(in_channels*4, in_channels*8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels*8, in_channels*2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels*2, int(in_channels//2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels//2), train_data.msg.shape[1]-32)                   
        )
        

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src) , self.lin_dst(z_dst)],dim=-1)      
         
        h = self.lin_seq (h)
        
        return h

memory_dim = 100         # node state
time_dim = 100
embedding_dim = 200      # edge embedding

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


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

saved_nodes=set()


# In[11]:


BATCH=1024
def train(train_data):

    
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    saved_nodes=set()

    total_loss = 0
    
#     print("train_before_stage_data:",train_data)
    for batch in train_data.seq_batches(batch_size=BATCH):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg        
        
        n_id = torch.cat([src, pos_dst]).unique()
#         n_id = torch.cat([src, pos_dst, neg_src, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
      
        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])       

        y_pred = torch.cat([pos_out], dim=0)
        
#         y_true = torch.cat([torch.zeros(pos_out.size(0),1),torch.ones(neg_out.size(0),1)], dim=0)#
        y_true=[]
        for m in msg:
            l=tensor_find(m[16:-16],1)-1
            y_true.append(l)           
          
        y_true = torch.tensor(y_true).to(device=device)
        y_true=y_true.reshape(-1).to(torch.long).to(device=device)
        
        loss = criterion(y_pred, y_true)
        
#         loss = criterion(pos_out, torch.ones_like(pos_out))
#         loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        
#         for i in range(len(src)):
#             saved_nodes.add(int(src[i]))
#             saved_nodes.add(int(pos_dst[i]))

        loss.backward()
        optimizer.step()
        memory.detach()
#         print(z.shape)
        total_loss += float(loss) * batch.num_events
#     print("trained_stage_data:",train_data)
    return total_loss / train_data.num_events



# In[12]:


train_graphs=[graph_5_8,graph_5_9,graph_5_11]

for epoch in tqdm(range(1, 31)):
    for g in train_graphs:
        loss = train(g)
        print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
#     scheduler.step()

model=[memory,gnn, link_pred,neighbor_loader]
os.system("mkdir -p ./models/")
torch.save(model,"./models/model_saved_share.pt")


# In[14]:


ls models/


# In[ ]:





# In[ ]:





# In[ ]:





# # Generate the reconstruction results of every day

# In[16]:


import time 

@torch.no_grad()
def test_day_new(inference_data,path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
    memory.eval()
    gnn.eval()
    link_pred.eval()
    
    memory.reset_state()  # Start with a fresh memory. 
    neighbor_loader.reset_state()  # Start with an empty graph.
    
    time_with_loss={} # key: time，  value： the losses
    total_loss = 0    
    edge_list=[]
    
    unique_nodes=torch.tensor([]).to(device=device)
    total_edges=0


    start_time=inference_data.t[0]
    event_count=0
    
    pos_o=[]
    
    loss_list=[]
    

    print("after merge:",inference_data)
    
    # Record the running time to evaluate the performance
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
#         y_true = torch.cat(
#             [torch.ones(pos_out.size(0))], dim=0).to(torch.long)     
#         y_true=y_true.reshape(-1).to(torch.long)

        y_true=[]
        for m in msg:
            l=tensor_find(m[16:-16],1)-1
            y_true.append(l) 
        y_true = torch.tensor(y_true).to(device=device)
        y_true=y_true.reshape(-1).to(torch.long).to(device=device)

        # Only consider which edge hasn't been correctly predicted.
        # For benign graphs, the behaviors patterns are similar and therefore their losses are small
        # For anoamlous behaviors, some behaviors might not be seen before, so the probability of predicting those edges are low. Thus their losses are high.
        loss = criterion(y_pred, y_true)

        total_loss += float(loss) * batch.num_events
     
        
        # update the edges in the batch to the memory and neighbor_loader
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        
        # compute the loss for each edge
        each_edge_loss= cal_pos_edges_loss_multiclass(pos_out,y_true)
        
        for i in range(len(pos_out)):
            srcnode=int(src[i])
            dstnode=int(pos_dst[i])  
            
            srcmsg=str(nodeid2msg[srcnode]) 
            dstmsg=str(nodeid2msg[dstnode])
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
            

#             if "netflow" in srcmsg or "netflow" in dstmsg:
#                 temp_dic['loss']=0
            edge_list.append(temp_dic)
        
        event_count+=len(batch.src)
        if t[-1]>start_time+60000000000*15:
            # Here is a checkpoint, which records all edge losses in the current time window
#             loss=total_loss/event_count
            time_interval=ns_time_to_datetime_US(start_time)+"~"+ns_time_to_datetime_US(t[-1])

            end = time.perf_counter()
            time_with_loss[time_interval]={'loss':loss,
                                
                                          'nodes_count':len(unique_nodes),
                                          'total_edges':total_edges,
                                          'costed_time':(end-start)}
            
            
            log=open(path+"/"+time_interval+".txt",'w')
            
            for e in edge_list: 
#                 temp_key=e['srcmsg']+e['dstmsg']+e['edge_type']
#                 if temp_key in train_edge_set:      
# #                     e['loss']=(e['loss']-train_edge_set[temp_key]) if e['loss']>=train_edge_set[temp_key] else 0  
# #                     e['loss']=abs(e['loss']-train_edge_set[temp_key])
                    
#                     e['modified']=True
#                 else:
#                     e['modified']=False
                loss+=e['loss']

            loss=loss/event_count   
            print(f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Cost Time: {(end-start):.2f}s')
            edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True)   # Rank the results based on edge losses
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



# In[15]:


graph_5_12=torch.load("./train_graph/graph_5_12.TemporalData.simple").to(device=device)
graph_5_15=torch.load("./train_graph/graph_5_15.TemporalData.simple").to(device=device)
graph_5_16=torch.load("./train_graph/graph_5_16.TemporalData.simple").to(device=device)
graph_5_17=torch.load("./train_graph/graph_5_17.TemporalData.simple").to(device=device)


# In[ ]:


model=torch.load("./models/model_saved_share.pt")
memory,gnn, link_pred,neighbor_loader=model


# In[17]:


ans_5_8=test_day_new(graph_5_8,"./graph_5_8")


# In[18]:


ans_5_9=test_day_new(graph_5_9,"./graph_5_9")


# In[19]:


ans_5_11=test_day_new(graph_5_11,"./graph_5_11")


# In[20]:


ans_5_12=test_day_new(graph_5_12,"./graph_5_12")


# In[21]:


ans_5_15=test_day_new(graph_5_15,"./graph_5_15")


# In[22]:


ans_5_16=test_day_new(graph_5_16,"./graph_5_16")


# In[23]:


ans_5_17=test_day_new(graph_5_17,"./graph_5_17")


# # Initialize the node IDF

# In[9]:


file_list=[]

file_path="graph_5_8/"
file_l=os.listdir("graph_5_8/")
for i in file_l:
    file_list.append(file_path+i)

file_path="graph_5_9/"
file_l=os.listdir("graph_5_9/")
for i in file_l:
    file_list.append(file_path+i)

file_path="graph_5_11/"
file_l=os.listdir("graph_5_11/")
for i in file_l:
    file_list.append(file_path+i)


file_path="graph_5_12/"
file_l=os.listdir("graph_5_12/")
for i in file_l:
    file_list.append(file_path+i)

# for f_path in tqdm(file_list):
#     f=open(f_path)
#     for line in f:
#         l=line.strip()
#         jdata=eval(l)
#         if jdata['loss']>0:
#             if 'netflow' not in str(jdata['srcmsg']):
#                 node_set.add(str(jdata['srcmsg']))
#             if 'netflow' not in str(jdata['dstmsg']):
#                 node_set.add(str(jdata['dstmsg'])) 

node_IDF={}
node_set = {}
for f_path in tqdm(file_list):
    f=open(f_path)
    for line in f:
        l=line.strip()
        jdata=eval(l)
        jdata=eval(l)
        if jdata['loss']>0:
            if 'netflow' not in str(jdata['srcmsg']):
                if str(jdata['srcmsg']) not in node_set.keys():
                    node_set[str(jdata['srcmsg'])] = set([f_path])
                else:
                    node_set[str(jdata['srcmsg'])].add(f_path)
            if 'netflow' not in str(jdata['dstmsg']):
                if str(jdata['dstmsg']) not in node_set.keys():
                    node_set[str(jdata['dstmsg'])] = set([f_path])
                else:
                    node_set[str(jdata['dstmsg'])].add(f_path)
for n in node_set:
    include_count = len(node_set[n])   
    IDF=math.log(len(file_list)/(include_count+1))
    node_IDF[n] = IDF    


torch.save(node_IDF,"node_IDF")
print("IDF weight calculate complete!")


# In[ ]:





# In[4]:


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
    
    thr=loss_mean+1.5*loss_std

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


# In[ ]:





# In[ ]:





# # Construct the relations between time windows

# In[12]:


# # node_IDF_5_9_12=torch.load("node_IDF_5_9-12")
# def cal_set_rel(s1,s2,file_list):
#     new_s=s1 & s2
#     count=0
#     for i in new_s:
# #     jdata=json.loads(i)
#         if 'netflow' not in i and '/home/admin/' not in i and  '/proc/' not in i :
        
# #         'netflow' not in i
# #         and 'usr' not in i and 'var' not in i
#             if i in node_IDF.keys():
#                 IDF=node_IDF[i]
#             else:
#                 IDF=math.log(len(file_list)/(1))
                
#             if i in node_IDF_4_4_7.keys():
#                 IDF4=node_IDF_4_4_7[i]
#             else:
#                 IDF4=math.log(len(file_list_4_4_7)/(1))    
            
# #             print(IDF)
#             if (IDF+IDF4)>9:
#                 print("node:",i," IDF:",IDF)
#                 count+=1
#     return count


# def cal_set_rel_bak(s1,s2,file_list):
#     new_s=s1 & s2
#     count=0
#     for i in new_s:
# #     jdata=json.loads(i)
#         if 'netflow' not in i \
#     and '/home/admin/' not in i \
#             and '/home/user/' not in i\
#             and '/tmp/' not in i\
#     and '/tmp/refload_pStageMem_log' not in i\
#             and 'com.' not in i:
        
# #         and '.dziauz.tinyflashlight' not in i \
# #             and '/data/system_ce/ not in i \
            
# #         and 'usr' not in i and 'proc' not in i and '675' not in i and 'firefox' not in i and 'tmp' not in i and 'thunderbird' not in i
# #         'netflow' not in i
# #         and 'usr' not in i and 'var' not in i
#             if i in node_IDF.keys():
#                 IDF=node_IDF[i]
#             else:
#                 IDF=math.log(len(file_list)/(1))           
                   
# #             print(IDF)
# #             print(len(file_list))
#             if IDF>math.log(len(file_list)*0.9/(1))  :
#                 print("node:",i," IDF:",IDF)
#                 count+=1
#     return count




def is_include_key_word_bak(s):
    keywords=[
         'netflow',

        '/home/admin/',
        '/home/user/',
         'proc',
        '/tmp/',
        '/var/spool/mqueue/',
        '/var/log/debug.log.0',
      
      ]
    flag=False
    for i in keywords:
        if i in s:
            flag=True
    return flag


def cal_set_rel_bak(s1,s2,file_list):
    new_s=s1 & s2
    count=0
    for i in new_s:
        if is_include_key_word_bak(i) is not True:
            if i in node_IDF.keys():
                IDF=node_IDF[i]
            else:
                IDF=math.log(len(file_list)/(1))           
                   
            if (IDF)>math.log(len(file_list)*0.9/(1))  :
                print("node:",i," IDF:",IDF)
                count+=1
    return count


# In[ ]:





# In[6]:


pred_label={}   

filelist = os.listdir("./graph_5_15")
for f in filelist:

    pred_label["./graph_5_15/"+f]=0
    
filelist = os.listdir("./graph_5_16")
for f in filelist:

    pred_label["./graph_5_16/"+f]=0
    
filelist = os.listdir("./graph_5_17")
for f in filelist:
    pred_label["./graph_5_17/"+f]=0


# In[7]:


pred_label


# # Anomaly Detection 5-15

# In[13]:


# node_IDF=torch.load("node_IDF_5_15")
node_IDF=torch.load("node_IDF")
y_data_5_15=[]
df_list_5_15=[]
# node_set_list=[]
history_list_5_15=[]
tw_que=[]
his_tw={}
current_tw={}
loss_list_5_15=[]

file_path_list=[]

file_path="./graph_5_15/"
file_l=os.listdir("./graph_5_15/")
for i in file_l:
    file_path_list.append(file_path+i)

index_count=0
for f_path in sorted(file_path_list):
    f=open(f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    # Figure out which nodes are anomalous in this time window
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
    df_list_5_15.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,"./graph_5_15/")

    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    # Incrementally construct the queues
    added_que_flag=False
    for hq in history_list_5_15:
        for his_tw in hq:
            if cal_set_rel_bak(current_tw['nodeset'],his_tw['nodeset'],file_list)!=0 and current_tw['name']!=his_tw['name']:
                print("history queue:",his_tw['name'])
                # check if there are intersection between two time windows.
                hq.append(copy.deepcopy(current_tw))
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list_5_15.append(temp_hq)
    index_count+=1
    loss_list_5_15.append(loss_avg)
    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))


# In[15]:


name_list=[]
for hl in history_list_5_15:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
#     name_list=[]
    if loss_count>100:
        name_list=[]
        for i in hl:
            name_list.append(i['name']) 
        print(name_list)
        for i in name_list:
            pred_label[i]=1
        print(loss_count)


# In[ ]:





# In[ ]:





# # Anoamly Detection 5-16

# In[16]:


# node_IDF=torch.load("node_IDF_5_16")
# node_IDF=torch.load("node_IDF_5_9-12")
y_data_5_16=[]
df_list_5_16=[]
# node_set_list=[]
history_list_5_16=[]
tw_que=[]
his_tw={}
current_tw={}

file_path_list=[]

file_path="./graph_5_16/"
file_l=os.listdir("./graph_5_16/")
for i in file_l:
    file_path_list.append(file_path+i)

index_count=0
for f_path in sorted(file_path_list):
    f=open(f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    # Figure out which nodes are anomalous in this time window
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
    df_list_5_16.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,"./graph_5_16/")

    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    # Incrementally construct the queues
    added_que_flag=False
    for hq in history_list_5_16:
        for his_tw in hq:
            if cal_set_rel_bak(current_tw['nodeset'],his_tw['nodeset'],file_list)!=0 and current_tw['name']!=his_tw['name']:
                print("history queue:",his_tw['name'])
                # check if there are intersection between two time windows.
                hq.append(copy.deepcopy(current_tw))
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list_5_16.append(temp_hq)
    index_count+=1
    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))


# In[19]:


name_list=[]
for hl in history_list_5_16:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
#     name_list=[]
    if loss_count>100:
        name_list=[]
        for i in hl:
            name_list.append(i['name'])
            print(i['name'])
#         print(name_list)
        for i in name_list:
            pred_label[i]=1
        print(loss_count)


# # Anomaly Detection 5-17

# In[20]:


# node_IDF=torch.load("node_IDF_5_17")
# node_IDF=torch.load("node_IDF_5_9-12")
y_data_5_17=[]
df_list_5_17=[]
# node_set_list=[]
history_list_5_17=[]
tw_que=[]
his_tw={}
current_tw={}

loss_list_5_17=[]

file_path_list=[]

file_path="./graph_5_17/"
file_l=os.listdir("./graph_5_17/")
for i in file_l:
    file_path_list.append(file_path+i)

index_count=0
for f_path in sorted(file_path_list):
    f=open(f_path)
    edge_loss_list=[]
    edge_list=[]
    print('index_count:',index_count)
    
    for line in f:
        l=line.strip()
        jdata=eval(l)
        edge_loss_list.append(jdata['loss'])
        edge_list.append([str(jdata['srcmsg']),str(jdata['dstmsg'])])
    df_list_5_17.append(pd.DataFrame(edge_loss_list))
    count,loss_avg,node_set,edge_set=cal_anomaly_loss(edge_loss_list,edge_list,"./clear_data/graph_5_17/")

    current_tw['name']=f_path
    current_tw['loss']=loss_avg
    current_tw['index']=index_count
    current_tw['nodeset']=node_set

    added_que_flag=False
    for hq in history_list_5_17:
        for his_tw in hq:
            if cal_set_rel_bak(current_tw['nodeset'],his_tw['nodeset'],file_list)!=0 and current_tw['name']!=his_tw['name']:
                print("history queue:",his_tw['name'])

                hq.append(copy.deepcopy(current_tw))
                added_que_flag=True
                break
            if added_que_flag:
                break
    if added_que_flag is False:
        temp_hq=[copy.deepcopy(current_tw)]
        history_list_5_17.append(temp_hq)
    index_count+=1
    loss_list_5_17.append(loss_avg)
    print( f_path,"  ",loss_avg," count:",count," percentage:",count/len(edge_list)," node count:",len(node_set)," edge count:",len(edge_set))


# In[22]:


name_list=[]
for hl in history_list_5_17:
    loss_count=0
    for hq in hl:
        if loss_count==0:
            loss_count=(loss_count+1)*(hq['loss']+1)
        else:
            loss_count=(loss_count)*(hq['loss']+1)
#     name_list=[]
    if loss_count>100:
        name_list=[]
        for i in hl:
            name_list.append(i['name']) 
            print(i['name'])
        for i in name_list:
            pred_label[i]=1
        print(loss_count)


# In[ ]:





# In[ ]:





# In[23]:


from sklearn.metrics import average_precision_score, roc_auc_score

# from sklearn.metrics import plot_roc_curve,roc_curve,auc,roc_auc_score
import torch
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

def plot_thr():
    np.seterr(invalid='ignore')
    step=0.01
    thr_list=torch.arange(-5,5,step)
    
    

    precision_list=[]
    recall_list=[]
    fscore_list=[]
    accuracy_list=[]
    auc_val_list=[]
    for thr in thr_list:
        threshold=thr
        y_prediction=[]
        for i in y_test_scores:
            if i >threshold:
                y_prediction.append(1)
            else:
                y_prediction.append(0)
        precision,recall,fscore,accuracy,auc_val=classifier_evaluation(y_test, y_prediction)   
        precision_list.append(float(precision))
        recall_list.append(float(recall))
        fscore_list.append(float(fscore))
        accuracy_list.append(float(accuracy))
        auc_val_list.append(float(auc_val))

    max_fscore=max(fscore_list)
    max_fscore_index=fscore_list.index(max_fscore)
    print(max_fscore_index)
    print("max threshold:",thr_list[max_fscore_index])
    print('precision:',precision_list[max_fscore_index])
    print('recall:',recall_list[max_fscore_index])
    print('fscore:',fscore_list[max_fscore_index])
    print('accuracy:',accuracy_list[max_fscore_index])    
    print('auc:',auc_val_list[max_fscore_index])

    plt.plot(thr_list,precision_list,color='red',label='precision',linewidth=2.0,linestyle='-')
    plt.plot(thr_list,recall_list,color='orange',label='recall',linewidth=2.0,linestyle='solid')
    plt.plot(thr_list,fscore_list,color='y',label='F-score',linewidth=2.0,linestyle='dashed')
    plt.plot(thr_list,accuracy_list,color='g',label='accuracy',linewidth=2.0,linestyle='dashdot')
    plt.plot(thr_list,auc_val_list,color='b',label='auc_val',linewidth=2.0,linestyle='dotted')


    plt.xlabel("Threshold", fontdict={'size': 16})
    plt.ylabel("Rate", fontdict={'size': 16})
    plt.title("Different evaluation Indicators by varying threshold value", fontdict={'size': 12})
    plt.legend(loc='best', fontsize=12, markerscale=0.5)
    plt.show()

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

def minmax(data):
    min_val=min(data)
    max_val=max(data)
    ans=[]
    for i in data:
        ans.append((i-min_val)/(max_val-min_val))
    return ans



# In[ ]:





# # label generation

# In[25]:


labels={}

filelist = os.listdir("graph_5_15")
for f in filelist:
    labels["./graph_5_15/"+f]=0

    
filelist = os.listdir("graph_5_16")
for f in filelist:
    labels["./graph_5_16/"+f]=0

    
filelist = os.listdir("graph_5_17")
for f in filelist:
    labels["./graph_5_17/"+f]=0


# In[26]:


attack_list=[
    './graph_5_16/2019-05-16 09:20:32.093582942~2019-05-16 09:36:08.903494477.txt', 
 './graph_5_16/2019-05-16 09:36:08.903494477~2019-05-16 09:51:22.110949680.txt', 
 './graph_5_16/2019-05-16 09:51:22.110949680~2019-05-16 10:06:29.403713371.txt', 
 './graph_5_16/2019-05-16 10:06:29.403713371~2019-05-16 10:21:47.983513184.txt', 

# Here are the "fake" FP time windows described in Section 5.2 in the paper.
 './graph_5_16/2019-05-16 20:32:27.570220441~2019-05-16 20:48:38.072848659.txt', 
 './graph_5_16/2019-05-16 21:19:00.930018779~2019-05-16 21:34:46.231624861.txt', 
 './graph_5_16/2019-05-16 21:34:46.231624861~2019-05-16 21:49:46.992678639.txt', 
 './graph_5_16/2019-05-16 21:49:46.992678639~2019-05-16 22:06:14.950154813.txt', 
 './graph_5_16/2019-05-16 22:06:14.950154813~2019-05-16 22:21:40.662702391.txt', 
 './graph_5_16/2019-05-16 22:21:40.662702391~2019-05-16 22:36:45.602858389.txt', 
 './graph_5_16/2019-05-16 22:36:45.602858389~2019-05-16 22:51:51.220035024.txt', 
 './graph_5_16/2019-05-16 22:51:51.220035024~2019-05-16 23:07:16.890296254.txt', 
 './graph_5_16/2019-05-16 23:07:16.890296254~2019-05-16 23:22:54.052353000.txt',

    
    './graph_5_17/2019-05-17 10:02:11.321524261~2019-05-17 10:17:26.881636687.txt', 
 './graph_5_17/2019-05-17 10:17:26.881636687~2019-05-17 10:32:38.131495470.txt', 
 './graph_5_17/2019-05-17 10:32:38.131495470~2019-05-17 10:48:02.091564015.txt'
]


# 结合 Rao的文档，对GT文档进行补充。检测出的准确率非常高。

for i in attack_list:
    labels[i]=1


# In[61]:


labels


# In[ ]:





# In[27]:


y=[]
y_pred=[]
for i in labels:
    y.append(labels[i])
    y_pred.append(pred_label[i])


# In[28]:


classifier_evaluation(y,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# # Count the number of the attack edges

# In[1]:


def keyword_hit(line):
    attack_nodes=[
           'nginx',
#         '128.55.12.167',
#          '4.21.51.250',
#          'ocMain.py',
        'python',
#          '98.23.182.25',
#         '108.192.100.31',
        'hostname',
        'whoami',
#         'cat /etc/passwd',  
        ]
    flag=False
    for i in attack_nodes:
        if i in line:
            flag=True
            break
    return flag



files=[
    'graph_5_16/2019-05-16 09:20:32.093582942~2019-05-16 09:36:08.903494477.txt', 
 'graph_5_16/2019-05-16 09:36:08.903494477~2019-05-16 09:51:22.110949680.txt', 
 'graph_5_16/2019-05-16 09:51:22.110949680~2019-05-16 10:06:29.403713371.txt', 
 'graph_5_16/2019-05-16 10:06:29.403713371~2019-05-16 10:21:47.983513184.txt', 
    'graph_5_17/2019-05-17 10:02:11.321524261~2019-05-17 10:17:26.881636687.txt', 
 'graph_5_17/2019-05-17 10:17:26.881636687~2019-05-17 10:32:38.131495470.txt', 
 'graph_5_17/2019-05-17 10:32:38.131495470~2019-05-17 10:48:02.091564015.txt'
]


# In[4]:


attack_edge_count=0
for fpath in tqdm(files):
    f=open(fpath)
    for line in f:
        if keyword_hit(line):
            attack_edge_count+=1
print(attack_edge_count)


# In[ ]:





# # Visualization

# In[49]:


import os

from graphviz import Digraph
import networkx as nx
import datetime
import community.community_louvain as community_louvain
from tqdm import tqdm



# Some common path abstraction for visualization
replace_dic = {
    '/run/shm/':'/run/shm/*',
    #     '/home/admin/.cache/mozilla/firefox/pe11scpa.default/cache2/entries/':'/home/admin/.cache/mozilla/firefox/pe11scpa.default/cache2/entries/*',
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
    'UNNAMED':'UNNAMED *',
    '/usr/ports/':'/usr/ports/*',
    '/usr/home/user/test':'/usr/home/user/test/*',
    '/tmp//':'/tmp//*',
    '/home/admin/backup/':'/home/admin/backup/*',
    '/home/admin/./backup/':'/home/admin/./backup/*',
    '/usr/home/admin/./test/':'/usr/home/admin/./test/*',
    '/usr/home/admin/test/':'/usr/home/admin/test/*',
    '/home/admin/out':'/home/admin/out*',
}


def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    return path_name


# Users should manually put the detected anomalous time windows here
attack_list = [
    'graph_5_16/2019-05-16 09:20:32.093582942~2019-05-16 09:36:08.903494477.txt', 
 'graph_5_16/2019-05-16 09:36:08.903494477~2019-05-16 09:51:22.110949680.txt', 
 'graph_5_16/2019-05-16 09:51:22.110949680~2019-05-16 10:06:29.403713371.txt', 
 'graph_5_16/2019-05-16 10:06:29.403713371~2019-05-16 10:21:47.983513184.txt', 
    'graph_5_17/2019-05-17 10:02:11.321524261~2019-05-17 10:17:26.881636687.txt', 
 'graph_5_17/2019-05-17 10:17:26.881636687~2019-05-17 10:32:38.131495470.txt', 
 'graph_5_17/2019-05-17 10:32:38.131495470~2019-05-17 10:48:02.091564015.txt'
]

original_edges_count = 0
graphs = []
gg = nx.DiGraph()
count = 0
for path in tqdm(attack_list):
    if ".txt" in path:
        line_count = 0
        node_set = set()
        tempg = nx.DiGraph()
        f = open(path, "r")
        edge_list = []
        for line in f:
            count += 1
            l = line.strip()
            jdata = eval(l)
            edge_list.append(jdata)

        edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)
        original_edges_count += len(edge_list)

        loss_list = []
        for i in edge_list:
            loss_list.append(i['loss'])
        loss_mean = mean(loss_list)
        loss_std = std(loss_list)
        print(loss_mean)
        print(loss_std)
        thr = loss_mean + 1.5 * loss_std
        print("thr:", thr)
        for e in edge_list:
            if e['loss'] > thr:
                tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),
                               str(hashgen(replace_path_name(e['dstmsg']))))
                gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))), str(hashgen(replace_path_name(e['dstmsg']))),
                            loss=e['loss'], srcmsg=e['srcmsg'], dstmsg=e['dstmsg'], edge_type=e['edge_type'],
                            time=e['time'])


partition = community_louvain.best_partition(gg.to_undirected())

# Generate the candidate subgraphs based on community discovery results
communities = {}
max_partition = 0
for i in partition:
    if partition[i] > max_partition:
        max_partition = partition[i]
for i in range(max_partition + 1):
    communities[i] = nx.DiGraph()
for e in gg.edges:
    communities[partition[e[0]]].add_edge(e[0], e[1])
    communities[partition[e[1]]].add_edge(e[0], e[1])


# Define the attack nodes. They are **only be used to plot the colors of attack nodes and edges**.
# They won't change the detection results.
def attack_edge_flag(msg):
    attack_edge_type = [
        "'nginx'",
        "'cat'",
        "'scp'",
        "'find'",
        "'bash'",
        "/etc/passwd",
        "/usr/home/user/",
        "128.55.12.167",
        "4.21.51.250",
        "128.55.12.233",
    ]
    flag = False
    for i in attack_edge_type:
        if i in str(msg):
            flag = True
    return flag


# Plot and render candidate subgraph
os.system(f"mkdir -p ./graph_visual/")
graph_index = 0
for c in communities:
    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr['rankdir'] = 'LR'

    for e in communities[c].edges:
        try:
            temp_edge = gg.edges[e]
            srcnode = e['srcnode']
            dstnode = e['dstnode']
        except:
            pass

        if True:
            # source node
            if "'subject': '" in temp_edge['srcmsg']:
                src_shape = 'box'
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape = 'oval'
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape = 'diamond'
            if attack_edge_flag(temp_edge['srcmsg']):
                src_node_color = 'red'
            else:
                src_node_color = 'blue'
            dot.node(name=str(hashgen(replace_path_name(temp_edge['srcmsg']))), label=str(
                replace_path_name(temp_edge['srcmsg']) + str(
                    partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])), color=src_node_color,
                     shape=src_shape)

            # destination node
            if "'subject': '" in temp_edge['dstmsg']:
                dst_shape = 'box'
            elif "'file': '" in temp_edge['dstmsg']:
                dst_shape = 'oval'
            elif "'netflow': '" in temp_edge['dstmsg']:
                dst_shape = 'diamond'
            if attack_edge_flag(temp_edge['dstmsg']):
                dst_node_color = 'red'
            else:
                dst_node_color = 'blue'
            dot.node(name=str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=str(
                replace_path_name(temp_edge['dstmsg']) + str(
                    partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])), color=dst_node_color,
                     shape=dst_shape)

            if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']):
                edge_color = 'red'
            else:
                edge_color = 'blue'
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                     str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=temp_edge['edge_type'],
                     color=edge_color)

    dot.render(f'./graph_visual/subgraph_' + str(graph_index), view=False)
    graph_index += 1





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




