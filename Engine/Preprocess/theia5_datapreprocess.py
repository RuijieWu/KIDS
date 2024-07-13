#!/usr/bin/env python
# coding: utf-8

# In[6]:


import functools
import os
import json
import multiprocessing as mp
import re
import torch
from tqdm import tqdm
from torch_geometric.data import *
import threading
import networkx as nx
import math

filePath="/the/absolute/path/of/raw_log/"
filelist = os.listdir(filePath)
import hashlib
def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest() 


# In[7]:


filelist = ['ta1-theia-1-e5-official-1.bin.1.json',
 'ta1-theia-1-e5-official-1.bin.1.json.1',
 'ta1-theia-1-e5-official-1.bin.1.json.2',
 'ta1-theia-1-e5-official-1.bin.2.json',
 'ta1-theia-1-e5-official-1.bin.json',
 'ta1-theia-1-e5-official-1.bin.json.1',
 'ta1-theia-1-e5-official-1.bin.json.2',
 'ta1-theia-1-e5-official-1.json.1',
 'ta1-theia-1-e5-official-1.json.2',
 'ta1-theia-1-e5-official-1.json',
 'ta1-theia-1-e5-official-2.bin.10.json',
 'ta1-theia-1-e5-official-2.bin.10.json.1',
 'ta1-theia-1-e5-official-2.bin.10.json.2',
 'ta1-theia-1-e5-official-2.bin.11.json',
 'ta1-theia-1-e5-official-2.bin.11.json.1',
 'ta1-theia-1-e5-official-2.bin.11.json.2',
 'ta1-theia-1-e5-official-2.bin.12.json',
 'ta1-theia-1-e5-official-2.bin.12.json.1',
 'ta1-theia-1-e5-official-2.bin.12.json.2',
 'ta1-theia-1-e5-official-2.bin.13.json',
 'ta1-theia-1-e5-official-2.bin.13.json.1',
 'ta1-theia-1-e5-official-2.bin.13.json.2',
 'ta1-theia-1-e5-official-2.bin.14.json',
 'ta1-theia-1-e5-official-2.bin.14.json.1',
 'ta1-theia-1-e5-official-2.bin.14.json.2',
 'ta1-theia-1-e5-official-2.bin.15.json',
 'ta1-theia-1-e5-official-2.bin.15.json.1',
 'ta1-theia-1-e5-official-2.bin.15.json.2',
 'ta1-theia-1-e5-official-2.bin.16.json',
 'ta1-theia-1-e5-official-2.bin.16.json.1',
 'ta1-theia-1-e5-official-2.bin.16.json.2',
 'ta1-theia-1-e5-official-2.bin.17.json',
 'ta1-theia-1-e5-official-2.bin.17.json.1',
 'ta1-theia-1-e5-official-2.bin.17.json.2',
 'ta1-theia-1-e5-official-2.bin.18.json',
 'ta1-theia-1-e5-official-2.bin.18.json.1',
 'ta1-theia-1-e5-official-2.bin.18.json.2',
 'ta1-theia-1-e5-official-2.bin.19.json',
 'ta1-theia-1-e5-official-2.bin.19.json.1',
 'ta1-theia-1-e5-official-2.bin.19.json.2',
 'ta1-theia-1-e5-official-2.bin.1.json',
 'ta1-theia-1-e5-official-2.bin.1.json.1',
 'ta1-theia-1-e5-official-2.bin.1.json.2',
 'ta1-theia-1-e5-official-2.bin.20.json',
 'ta1-theia-1-e5-official-2.bin.20.json.1',
 'ta1-theia-1-e5-official-2.bin.20.json.2',
 'ta1-theia-1-e5-official-2.bin.21.json',
 'ta1-theia-1-e5-official-2.bin.21.json.1',
 'ta1-theia-1-e5-official-2.bin.21.json.2',
 'ta1-theia-1-e5-official-2.bin.22.json',
 'ta1-theia-1-e5-official-2.bin.22.json.1',
 'ta1-theia-1-e5-official-2.bin.22.json.2',
 'ta1-theia-1-e5-official-2.bin.23.json',
 'ta1-theia-1-e5-official-2.bin.23.json.1',
 'ta1-theia-1-e5-official-2.bin.23.json.2',
 'ta1-theia-1-e5-official-2.bin.24.json',
 'ta1-theia-1-e5-official-2.bin.24.json.1',
 'ta1-theia-1-e5-official-2.bin.24.json.2',
 'ta1-theia-1-e5-official-2.bin.25.json',
 'ta1-theia-1-e5-official-2.bin.25.json.1',
 'ta1-theia-1-e5-official-2.bin.25.json.2',
 'ta1-theia-1-e5-official-2.bin.26.json',
 'ta1-theia-1-e5-official-2.bin.26.json.1',
 'ta1-theia-1-e5-official-2.bin.26.json.2',
 'ta1-theia-1-e5-official-2.bin.27.json',
 'ta1-theia-1-e5-official-2.bin.27.json.1',
 'ta1-theia-1-e5-official-2.bin.27.json.2',
 'ta1-theia-1-e5-official-2.bin.28.json',
 'ta1-theia-1-e5-official-2.bin.28.json.1',
 'ta1-theia-1-e5-official-2.bin.28.json.2',
 'ta1-theia-1-e5-official-2.bin.29.json',
 'ta1-theia-1-e5-official-2.bin.29.json.1',
 'ta1-theia-1-e5-official-2.bin.29.json.2',
 'ta1-theia-1-e5-official-2.bin.2.json',
 'ta1-theia-1-e5-official-2.bin.2.json.1',
 'ta1-theia-1-e5-official-2.bin.2.json.2',
 'ta1-theia-1-e5-official-2.bin.30.json',
 'ta1-theia-1-e5-official-2.bin.30.json.1',
 'ta1-theia-1-e5-official-2.bin.30.json.2',
 'ta1-theia-1-e5-official-2.bin.31.json',
 'ta1-theia-1-e5-official-2.bin.31.json.1',
 'ta1-theia-1-e5-official-2.bin.31.json.2',
 'ta1-theia-1-e5-official-2.bin.32.json',
 'ta1-theia-1-e5-official-2.bin.32.json.1',
 'ta1-theia-1-e5-official-2.bin.32.json.2',
 'ta1-theia-1-e5-official-2.bin.33.json',
 'ta1-theia-1-e5-official-2.bin.33.json.1',
 'ta1-theia-1-e5-official-2.bin.33.json.2',
 'ta1-theia-1-e5-official-2.bin.34.json',
 'ta1-theia-1-e5-official-2.bin.34.json.1',
 'ta1-theia-1-e5-official-2.bin.34.json.2',
 'ta1-theia-1-e5-official-2.bin.35.json',
 'ta1-theia-1-e5-official-2.bin.35.json.1',
 'ta1-theia-1-e5-official-2.bin.35.json.2',
 'ta1-theia-1-e5-official-2.bin.36.json',
 'ta1-theia-1-e5-official-2.bin.36.json.1',
 'ta1-theia-1-e5-official-2.bin.36.json.2',
 'ta1-theia-1-e5-official-2.bin.37.json',
 'ta1-theia-1-e5-official-2.bin.37.json.1',
 'ta1-theia-1-e5-official-2.bin.37.json.2',
 'ta1-theia-1-e5-official-2.bin.38.json',
 'ta1-theia-1-e5-official-2.bin.38.json.1',
 'ta1-theia-1-e5-official-2.bin.38.json.2',
 'ta1-theia-1-e5-official-2.bin.39.json',
 'ta1-theia-1-e5-official-2.bin.3.json',
 'ta1-theia-1-e5-official-2.bin.3.json.1',
 'ta1-theia-1-e5-official-2.bin.3.json.2',
 'ta1-theia-1-e5-official-2.bin.4.json',
 'ta1-theia-1-e5-official-2.bin.4.json.1',
 'ta1-theia-1-e5-official-2.bin.4.json.2',
 'ta1-theia-1-e5-official-2.bin.5.json',
 'ta1-theia-1-e5-official-2.bin.5.json.1',
 'ta1-theia-1-e5-official-2.bin.5.json.2',
 'ta1-theia-1-e5-official-2.bin.6.json',
 'ta1-theia-1-e5-official-2.bin.6.json.1',
 'ta1-theia-1-e5-official-2.bin.6.json.2',
 'ta1-theia-1-e5-official-2.bin.7.json',
 'ta1-theia-1-e5-official-2.bin.7.json.1',
 'ta1-theia-1-e5-official-2.bin.7.json.2',
 'ta1-theia-1-e5-official-2.bin.8.json',
 'ta1-theia-1-e5-official-2.bin.8.json.1',
 'ta1-theia-1-e5-official-2.bin.8.json.2',
 'ta1-theia-1-e5-official-2.bin.9.json',
 'ta1-theia-1-e5-official-2.bin.9.json.1',
 'ta1-theia-1-e5-official-2.bin.9.json.2',
 'ta1-theia-1-e5-official-2.bin.json',
 'ta1-theia-1-e5-official-2.bin.json.1',
 'ta1-theia-1-e5-official-2.bin.json.2']


# In[8]:


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


# In[ ]:





# # Database setting (Make sure the database and tables are created)

# In[9]:


import psycopg2

from psycopg2 import extras as ex
connect = psycopg2.connect(database = 'tc_e5_theia_dataset_db',
                           host = '/var/run/postgresql/',
                           user = 'postgres',
                           password = 'postgres',
                           port = '5432'
                          )

cur = connect.cursor()


# In[ ]:





# ## Netflow

# In[9]:


netobjset=set()
netobj2hash={}# 
datalist=[]
for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
            for line in f:
#                 pass
                if "NetFlowObject" in line:
#                     print(line)
                    try:
                        res=re.findall('NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}',line)[0]

                        nodeid=res[0]
                        srcaddr=res[2]
                        srcport=res[3]
                        dstaddr=res[4]
                        dstport=res[5]

                        nodeproperty=srcaddr+","+srcport+","+dstaddr+","+dstport 
#                         nodeproperty=dstaddr+","+dstport 
                        hashstr=stringtomd5(nodeproperty)
                        netobj2hash[nodeid]=[hashstr,nodeproperty]
                        netobj2hash[hashstr]=nodeid
                        netobjset.add(hashstr)
                    except:
                        pass
#                     print(match)


# In[10]:


datalist=[]
for i in netobj2hash.keys():
    if len(i)!=64:
        datalist.append([i]+[netobj2hash[i][0]]+netobj2hash[i][1].split(","))

sql = '''insert into netflow_node_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist,page_size=10000)
connect.commit() 

# del netobj2hash
# del datalist


# In[ ]:





# ## Process

# In[11]:


scusess_count=0
fail_count=0
subject_objset=set()
subject_obj2hash={}# 
for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
#             for line in tqdm(f): 
            for line in (f):
                if "schema.avro.cdm20.Subject" in line:
#                     print(line)
                    subject_uuid=re.findall('avro.cdm20.Subject":{"uuid":"(.*?)",(.*?)"path":"(.*?)"',line)
#                
                    try:
#                         (subject_uuid[0][-1])
                        subject_obj2hash[subject_uuid[0][0]]=subject_uuid[0][-1]
                        scusess_count+=1
                    except:
                        try:
                            subject_obj2hash[subject_uuid[0][0]]="null"
                        except:
                            pass
#                             print(line)
#                         print(line)                        
                        fail_count+=1
                        


# In[16]:


datalist=[]
for i in subject_obj2hash.keys():
    if len(i)!=64:
        datalist.append([i]+[stringtomd5(subject_obj2hash[i]),subject_obj2hash[i]])


sql = '''insert into subject_node_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist,page_size=10000)
connect.commit() 


# del netobj2hash
# del datalist


# In[17]:


datalist


# ## File

# In[5]:


file_node=set()
file_obj2hash={}
fail_count=0
for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
            for line in f:
                if "avro.cdm20.FileObject" in line:
#                     print(line)
                    Object_uuid=re.findall('avro.cdm20.FileObject":{"uuid":"(.*?)",(.*?)"filename":"(.*?)"',line) 
                    try:
                        file_node.add(Object_uuid[0])
                        file_obj2hash[Object_uuid[0][0]]=Object_uuid[0][-1]
                    except:
                        fail_count+=1
#                         print(line)


# In[6]:


datalist=[]
for i in file_obj2hash.keys():
    if len(i)!=64:
        datalist.append([i]+[stringtomd5(file_obj2hash[i]),file_obj2hash[i]])

sql = '''insert into file_node_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist,page_size=10000)
connect.commit() 


# ## Processing the event data

# In[7]:


# Generate the data for node2id table
node_list={}
##################################################################################################
sql="""
select * from file_node_table;
"""
cur.execute(sql)
records = cur.fetchall()

for i in records:    
    node_list[i[1]]=["file",i[-1]]

file_uuid2hash={}
for i in records:
    file_uuid2hash[i[0]]=i[1]
##################################################################################################    
sql="""
select * from subject_node_table;
"""
cur.execute(sql)
records = cur.fetchall()

for i in records:
    node_list[i[1]]=["subject",i[-1]]

subject_uuid2hash={}
for i in records:
    subject_uuid2hash[i[0]]=i[1]
##################################################################################################
sql="""
select * from netflow_node_table;
"""
cur.execute(sql)
records = cur.fetchall()

for i in records:
    
    node_list[i[1]]=["netflow",i[-2]+":"+i[-1]]

net_uuid2hash={}
for i in records:
    net_uuid2hash[i[0]]=i[1]


# In[8]:


node_list_database=[]
node_index=0
for i in node_list:
    node_list_database.append([i]+node_list[i]+[node_index])
    node_index+=1


# In[9]:


sql = '''insert into node2id
                     values %s
        '''
ex.execute_values(cur,sql, node_list_database,page_size=10000)
connect.commit()  


# In[10]:


# Constructing the map for nodeid to msg
sql="select * from node2id ORDER BY index_id;"
cur.execute(sql)
rows = cur.fetchall()

nodeid2msg={}  # nodeid => msg and node hash => nodeid
for i in rows:
    nodeid2msg[i[0]]=i[-1]
    nodeid2msg[i[-1]]={i[1]:i[2]}


# In[11]:


nodeid2msg


# In[12]:


include_edge_type=[
    'EVENT_CLOSE',
    'EVENT_OPEN',
    'EVENT_READ',
    'EVENT_WRITE',
     'EVENT_EXECUTE',
    'EVENT_RECVFROM',
    'EVENT_RECVMSG',
    'EVENT_SENDMSG',
    'EVENT_SENDTO',
]


# In[13]:


def write_event_in_DB(datalist_new):
    sql = '''insert into event_table
                         values %s
            '''
    ex.execute_values(cur,sql, datalist_new,page_size=10000)
    connect.commit() 


# In[14]:


datalist=[]
edge_type=set()
reverse=["EVENT_RECVFROM","EVENT_RECVMSG","EVENT_READ"]        
for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
            for line in (f):
                if '{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line:
#                     print(line)
                    subject_uuid=re.findall('"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"',line)
                    predicateObject_uuid=re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"',line)
                    if len(subject_uuid) >0 and len(predicateObject_uuid)>0:
                        if subject_uuid[0] in subject_uuid2hash\
                        and (predicateObject_uuid[0] in file_uuid2hash or predicateObject_uuid[0] in net_uuid2hash):
                            relation_type=re.findall('"type":"(.*?)"',line)[0]
                            time_rec=re.findall('"timestampNanos":(.*?),',line)[0]
                            time_rec=int(time_rec) 
                            subjectId=subject_uuid2hash[subject_uuid[0]]
                            if predicateObject_uuid[0] in file_uuid2hash:
                                objectId=file_uuid2hash[predicateObject_uuid[0]]
                            else:
                                objectId=net_uuid2hash[predicateObject_uuid[0]]
#                                 print(line)
                            edge_type.add(relation_type)
                            if relation_type in reverse:
                                datalist.append([objectId,nodeid2msg[objectId],relation_type,subjectId,nodeid2msg[subjectId],time_rec])
                            else:
                                datalist.append([subjectId,nodeid2msg[subjectId],relation_type,objectId,nodeid2msg[objectId],time_rec])

               
     
                     


# In[16]:


sql = '''insert into event_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist,page_size=50000)
connect.commit() 


# In[ ]:





# In[ ]:





# # Featurization

# In[17]:


from sklearn.feature_extraction import FeatureHasher
from torch_geometric.transforms import NormalizeFeatures

from sklearn import preprocessing 
import numpy as np


FH_string=FeatureHasher(n_features=16,input_type="string")
FH_dict=FeatureHasher(n_features=16,input_type="dict")


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
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
#     print(l)
    return l


def subject2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
#     print(l)
    return l


def list2str(l):
    s=''
    for i in l:
        s+=i
    return s


# In[18]:


node_msg_vec=[]
node_msg_dic_list=[]
for i in tqdm(nodeid2msg.keys()):
    if type(i)==int:
        if 'netflow' in nodeid2msg[i].keys():
            higlist=['netflow']
            higlist+=ip2higlist(nodeid2msg[i]['netflow'])
            
        if 'file' in nodeid2msg[i].keys():
            higlist=['file']
            higlist+=path2higlist(nodeid2msg[i]['file'])
            
#             print(higlist)
        if 'subject' in nodeid2msg[i].keys():
            higlist=['subject']
            higlist+=subject2higlist(nodeid2msg[i]['subject'])
        node_msg_dic_list.append(list2str(higlist))


# In[19]:


node2higvec=[]
for i in tqdm(node_msg_dic_list):
    vec=FH_string.transform([i]).toarray()
    node2higvec.append(vec)


# In[20]:


node2higvec=np.array(node2higvec).reshape([-1,16])


# In[12]:


rel2id={1: 'EVENT_CONNECT',
 'EVENT_CONNECT': 1,
 2: 'EVENT_EXECUTE',
 'EVENT_EXECUTE': 2,
 3: 'EVENT_OPEN',
 'EVENT_OPEN': 3,
 4: 'EVENT_READ',
 'EVENT_READ': 4,
 5: 'EVENT_RECVFROM',
 'EVENT_RECVFROM': 5,
 6: 'EVENT_RECVMSG',
 'EVENT_RECVMSG': 6,
 7: 'EVENT_SENDMSG',
 'EVENT_SENDMSG': 7,
 8: 'EVENT_SENDTO',
 'EVENT_SENDTO': 8,
 9: 'EVENT_WRITE',
 'EVENT_WRITE': 9}


# In[22]:


# Geneate edge type one-hot
relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)


# In[23]:


# Map different relation types to their one-hot encoding
rel2vec={}
for i in rel2id.keys():
    if type(i) is not int:
        rel2vec[i]= relvec[rel2id[i]-1]
        rel2vec[relvec[rel2id[i]-1]]=i


# In[24]:


## save the results
torch.save(node2higvec,"node2higvec")
torch.save(rel2vec,"rel2vec")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Generate the dataset

# In[10]:


node2higvec=torch.load("./node2higvec")
rel2vec=torch.load("./rel2vec")


# In[13]:


os.system("mkdir -p ./train_graphs/")
for day in tqdm(range(8,18)):
    start_timestamp=datetime_to_ns_time_US('2019-05-'+str(day)+' 00:00:00')
    end_timestamp=datetime_to_ns_time_US('2019-05-'+str(day+1)+' 00:00:00')
    sql="""
    select * from event_table
    where
          timestamp_rec>'%s' and timestamp_rec<'%s'
           ORDER BY timestamp_rec;
    """%(start_timestamp,end_timestamp)
    cur.execute(sql)
    events = cur.fetchall()
    print('2019-05-'+str(day)," events count:",str(len(events)))
    edge_list=[]
    for e in events:
        edge_temp=[int(e[1]),int(e[4]),e[2],e[5]]
        if e[2] in rel2id:
#         if True:
            edge_list.append(edge_temp)
    print('2019-05-'+str(day)," edge list len:",str(len(edge_list)))
    
    if len(edge_list) == 0:
        continue
    

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for i in edge_list:
        src.append(int(i[0]))
        dst.append(int(i[1]))
    #     msg.append(torch.cat([torch.from_numpy(node2higvec_bn[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec_bn[i[1]])] ))
        msg.append(torch.cat([torch.from_numpy(node2higvec[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec[i[1]])] ))
        t.append(int(i[3]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, "./train_graphs/graph_5_"+str(day)+".TemporalData.simple")  


# In[ ]:





# In[ ]:




