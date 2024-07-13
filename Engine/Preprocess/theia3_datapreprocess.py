#!/usr/bin/env python
# coding: utf-8

# In[1]:


import functools
import os
import json
import re
import torch
from tqdm import tqdm
from torch_geometric.data import *
import networkx as nx

import hashlib
def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()
filePath="/the/absolute/path/of/raw_log/"

filelist = os.listdir(filePath)


# In[2]:


sorted(filelist)


# In[10]:


# filelist=[
#     'ta1-theia-e3-official-1r.json',
#     'ta1-theia-e3-official-1r.json.1',
#     'ta1-theia-e3-official-1r.json.2',
#     'ta1-theia-e3-official-1r.json.3',
#     'ta1-theia-e3-official-1r.json.4',
#     'ta1-theia-e3-official-1r.json.5',
#     'ta1-theia-e3-official-1r.json.6',
#     'ta1-theia-e3-official-1r.json.7',
#     'ta1-theia-e3-official-1r.json.8',
#     'ta1-theia-e3-official-1r.json.9',
#     'ta1-theia-e3-official-3.json',
#     'ta1-theia-e3-official-5m.json',
#     'ta1-theia-e3-official-6r.json',
#     'ta1-theia-e3-official-6r.json.1',
#     'ta1-theia-e3-official-6r.json.2',
#     'ta1-theia-e3-official-6r.json.3',
#     'ta1-theia-e3-official-6r.json.4',
#     'ta1-theia-e3-official-6r.json.5',
#     'ta1-theia-e3-official-6r.json.6',
#     'ta1-theia-e3-official-6r.json.7',
#     'ta1-theia-e3-official-6r.json.8',
#     'ta1-theia-e3-official-6r.json.9',
#     'ta1-theia-e3-official-6r.json.10',
#     'ta1-theia-e3-official-6r.json.11',
#     'ta1-theia-e3-official-6r.json.12'
# ]
filelist = ['ta1-theia-e3-official-6r.json.9',
 'ta1-theia-e3-official-1r.json.9',
 'ta1-theia-e3-official-6r.json.8',
 'ta1-theia-e3-official-6r.json.12',
 'ta1-theia-e3-official-1r.json.7',
 'ta1-theia-e3-official-6r.json.7',
 'ta1-theia-e3-official-6r.json.5',
 'ta1-theia-e3-official-1r.json.3',
 'ta1-theia-e3-official-6r.json',
 'ta1-theia-e3-official-1r.json.5',
 'ta1-theia-e3-official-6r.json.11',
 'ta1-theia-e3-official-1r.json.4',
 'ta1-theia-e3-official-1r.json.6',
 'ta1-theia-e3-official-5m.json',
 'ta1-theia-e3-official-1r.json.2',
 'ta1-theia-e3-official-6r.json.10',
 'ta1-theia-e3-official-6r.json.4',
 'ta1-theia-e3-official-6r.json.1',
 'ta1-theia-e3-official-3.json',
 'ta1-theia-e3-official-1r.json.8',
 'ta1-theia-e3-official-1r.json.1',
 'ta1-theia-e3-official-6r.json.6',
 'ta1-theia-e3-official-6r.json.2',
 'ta1-theia-e3-official-6r.json.3',
 'ta1-theia-e3-official-1r.json']


# In[41]:


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

# In[5]:


import psycopg2

from psycopg2 import extras as ex
connect = psycopg2.connect(database = 'tc_theia_dataset_db',
                           host = '/var/run/postgresql/',
                           user = 'postgres',
                           password = 'postgres',
                           port = '5432'
                          )

cur = connect.cursor()


# In[ ]:





# In[ ]:





# ## Netflow

# In[12]:


netobjset=set()
netobj2hash={}# 
datalist=[]
for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
            for line in f:
#                 pass
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' in line :
            
                    try:
                        res=re.findall('NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',line)[0]

                        nodeid=res[0]
                        srcaddr=res[2]
                        srcport=res[3]
                        dstaddr=res[4]
                        dstport=res[5]

                        nodeproperty=srcaddr+","+srcport+","+dstaddr+","+dstport 
                        hashstr=stringtomd5(nodeproperty)
                        netobj2hash[nodeid]=[hashstr,nodeproperty]
                        netobj2hash[hashstr]=nodeid
                        netobjset.add(hashstr)
                    except:
                        
                        pass




# In[13]:


datalist=[]
for i in netobj2hash.keys():
    if len(i)!=64:
        datalist.append([i]+[netobj2hash[i][0]]+netobj2hash[i][1].split(","))



sql = '''insert into netflow_node_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist,page_size=10000)
connect.commit()  


# In[ ]:





# ## Process

# In[14]:


scusess_count=0
fail_count=0
subject_obj2hash={}# 

subjectset=set()
subject2hash={}

datalist=[]
for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line :
                    res=re.findall('Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"',line)[0]
                    try:
                        path_str=re.findall('"path":"(.*?)"',line)[0] 
                        path=path_str
                    except:
                        path="null"
                    nodeid=res[0]
                    cmdLine=res[2]
                    tgid=res[4]
                    
                    nodeproperty=cmdLine+","+tgid+","+path
#                     if len(cmdLine) > 1000:
#                         print(file)
#                         print(cmdLine)
#                         print(line)
#                         input()
                    
                    hashstr=stringtomd5(nodeproperty)
                    subject2hash[nodeid]=[hashstr,cmdLine,tgid,path]
                    subject2hash[hashstr]=nodeid
                    subjectset.add(hashstr)
                        


# In[15]:


subject2hash


# In[16]:


datalist=[]
for i in subject2hash.keys():
    if len(i)!=64:
        datalist.append([i]+subject2hash[i])


# In[17]:


datalist[0][2]


# In[18]:


i


# In[19]:


len(datalist)


# In[20]:


sql = '''insert into subject_node_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist,page_size=10000)
connect.commit()    


# In[ ]:





# ## File

# In[21]:


fileset=set()

# Remove the file nodes whose paths are null. So when parsing the edges, 
# if the src or dst node is file and it doesn't have pathname, then we can ignore them.

file2hash={}  
nullcount=0


for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line :
                    try:

                        res=re.findall('FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"',line)[0]
                        nodeid=res[0]
                        filepath=res[2]
                        nodeproperty=filepath
                        hashstr=stringtomd5(nodeproperty)
                        file2hash[nodeid]=[hashstr,nodeproperty]
                        file2hash[hashstr]=nodeid
                        fileset.add(hashstr)
                    except:
                        pass


# In[22]:


datalist=[]
for i in file2hash.keys():
    if len(i)!=64:
        datalist.append([i]+file2hash[i])


# In[23]:


sql = '''insert into file_node_table
                     values %s
        '''
ex.execute_values(cur,sql, datalist,page_size=10000)
connect.commit()  


# ## Processing the event data

# In[24]:


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


# In[25]:


node_list_database=[]
node_index=0
for i in node_list:
    node_list_database.append([i]+node_list[i]+[node_index])
    node_index+=1


# In[26]:


sql = '''insert into node2id
                     values %s
        '''
ex.execute_values(cur,sql, node_list_database,page_size=10000)
connect.commit() 


# In[27]:


# Constructing the map for nodeid to msg
sql="select * from node2id ORDER BY index_id;"
cur.execute(sql)
rows = cur.fetchall()

nodeid2msg={}  # nodeid => msg and node hash => nodeid
for i in rows:
    nodeid2msg[i[0]]=i[-1]
    nodeid2msg[i[-1]]={i[1]:i[2]} 


# In[28]:


nodeid2msg


# In[33]:


datalist=[]


# firsttime=1522764134000000000

for file in tqdm(filelist):
        with open(filePath + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line:
                    time=re.findall('"timestampNanos":(.*?),',line)[0]
                    time=int(time)
                    subjectid=re.findall('"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"',line)[0]
                    objectid=re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"',line)[0]
                    relation_type=re.findall('"type":"(.*?)"',line)[0]


                    if subjectid in subject2hash.keys():
                        # this subject is recorded
                        subjectid=subject2hash[subjectid][0] # convert uuid to hash value
                    if objectid in subject2hash.keys():
                        objectid=subject2hash[objectid][0] # convert uuid to hash value
                    if objectid in file2hash.keys():
                        objectid=file2hash[objectid][0] # convert uuid to hash value
                    if objectid in netobj2hash.keys():
                        objectid=netobj2hash[objectid][0] # convert uuid to hash value
                    if len(subjectid)==64 and len(objectid) == 64:
                        # Only when both src and dst UUIDs are converted into hashid will this edge be vaild and stored.
                        if relation_type in ['EVENT_READ','EVENT_READ_SOCKET_PARAMS','EVENT_RECVFROM','EVENT_RECVMSG']:
                            # data flows to subject, so process is dst node of the event
                            datalist.append((objectid,nodeid2msg[objectid],relation_type,subjectid,nodeid2msg[subjectid],time))
                        else:
                            datalist.append((subjectid,nodeid2msg[subjectid],relation_type,objectid,nodeid2msg[objectid],time))

                     
                        
                        if len(datalist)>=10000:
                            try:                
                                sql = '''insert into event_table
                                                     values %s
                                        '''
                                ex.execute_values(cur,sql, datalist,page_size=100000)
                                connect.commit()  
                                datalist=[]
                            except:
                                print("Failed to wirte database！")
                                connect.rollback()


# In[ ]:





# # Featurization

# In[30]:


from sklearn.feature_extraction import FeatureHasher
from torch_geometric.transforms import NormalizeFeatures

from sklearn import preprocessing 
import numpy as np
# 
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
def list2str(l):
    s=''
    for i in l:
        s+=i
    return s


# In[31]:


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
            higlist+=path2higlist(nodeid2msg[i]['subject'])
        node_msg_dic_list.append(list2str(higlist))


# In[32]:


node2higvec=[]
for i in tqdm(node_msg_dic_list):
    vec=FH_string.transform([i]).toarray()
    node2higvec.append(vec)


# In[34]:


node2higvec=np.array(node2higvec).reshape([-1,16])


# In[35]:


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


# In[36]:


# Geneate edge type one-hot
relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)


# In[37]:


# Map different relation types to their one-hot encoding
rel2vec={}
for i in rel2id.keys():
    if type(i) is not int:
        rel2vec[i]= relvec[rel2id[i]-1]
        rel2vec[relvec[rel2id[i]-1]]=i


# In[38]:


## save the results
torch.save(node2higvec,"node2higvec")
torch.save(rel2vec,"rel2vec")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Generate the dataset

# In[39]:


node2higvec=torch.load("./node2higvec")
rel2vec=torch.load("./rel2vec")


# In[46]:


os.system("mkdir -p ./train_graph/")
for day in tqdm(range(2,14)):
    start_timestamp=datetime_to_ns_time_US('2018-04-'+str(day)+' 00:00:00')
    end_timestamp=datetime_to_ns_time_US('2018-04-'+str(day+1)+' 00:00:00')
    sql="""
    select * from event_table
    where
          timestamp_rec>'%s' and timestamp_rec<'%s'
           ORDER BY timestamp_rec;
    """%(start_timestamp,end_timestamp)
    cur.execute(sql)
    events = cur.fetchall()
    print('2018-04-'+str(day)," events count:",str(len(events)))
    edge_list=[]
    for e in events:
        edge_temp=[int(e[1]),int(e[4]),e[2],e[5]]
        if e[2] in rel2id:
            edge_list.append(edge_temp)
    print('2018-04-'+str(day)," edge list len:",str(len(edge_list)))
    
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
    torch.save(dataset, "./train_graph/graph_4_"+str(day)+".TemporalData.simple")  


# In[ ]:





# In[ ]:





# In[ ]:




