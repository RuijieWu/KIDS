'''
Date: 2024-06-12 21:29:27
LastEditTime: 2024-06-13 19:16:30
Description: Embed Events from database into GNN
'''
from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm

import numpy as np
import torch

from config import *
from utils import *

#*     time: {
#*         events_count:len(events)
#*         edges_count:len(edge_list)
#*      }
#*
#* The are statics of the database

def gen_feature(cur,recording = False):
    # Firstly obtain all node labels
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Construct the hierarchical representation for each node label
    node_msg_dic_list = []
    for i in tqdm(nodeid2msg.keys()):
        if type(i) == int:
            if 'netflow' in nodeid2msg[i].keys():
                higlist = ['netflow']
                higlist += ip2higlist(nodeid2msg[i]['netflow'])

            if 'file' in nodeid2msg[i].keys():
                higlist = ['file']
                higlist += path2higlist(nodeid2msg[i]['file'])

            if 'subject' in nodeid2msg[i].keys():
                higlist = ['subject']
                higlist += path2higlist(nodeid2msg[i]['subject'])
            node_msg_dic_list.append(list2str(higlist))

    # Featurize the hierarchical node labels
    FH_string = FeatureHasher(n_features=NODE_EMBEDDING_DIM, input_type="string")
    node2higvec=[]
    for i in tqdm(node_msg_dic_list):
        vec=FH_string.transform([i]).toarray()
        node2higvec.append(vec)
    node2higvec = np.array(node2higvec).reshape([-1, NODE_EMBEDDING_DIM])
    if recording:
        torch.save(node2higvec, artifact_dir + "node2higvec")
    return node2higvec

def gen_relation_onehot(recording = False):
    relvec = torch.nn.functional.one_hot(torch.arange(0, len(REL2ID.keys())//2), num_classes=len(REL2ID.keys())//2)
    rel2vec = {}
    for i in REL2ID.keys():
        if type(i) is not int:
            rel2vec[i]= relvec[REL2ID[i]-1]
            rel2vec[relvec[REL2ID[i]-1]]=i
    if recording:
        torch.save(rel2vec, artifact_dir + "rel2vec")
    return rel2vec

def gen_vectorized_graphs(
    cur,
    node2higvec,
    rel2vec,
    recording = False
):
    graphs = []
    for day in tqdm(range(2, 14)):
        start_timestamp = datetime_to_ns_time_US('2018-04-' + str(day) + ' 00:00:00')
        end_timestamp = datetime_to_ns_time_US('2018-04-' + str(day + 1) + ' 00:00:00')
        sql = """
        select * from event_table
        where
              timestamp_rec>'%s' and timestamp_rec<'%s'
               ORDER BY timestamp_rec;
        """ % (start_timestamp, end_timestamp)
        cur.execute(sql)
        events = cur.fetchall()
        edge_list = []
        for e in events:
            edge_temp = [int(e[1]), int(e[4]), e[2], e[5]]
            if e[2] in EDGE_TYPE:
                edge_list.append(edge_temp)
        dataset = TemporalData()
        src = []
        dst = []
        msg = []
        t = []
        for i in edge_list:
            src.append(int(i[0]))
            dst.append(int(i[1]))
            msg.append(
                torch.cat([torch.from_numpy(node2higvec[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec[i[1]])]))
            t.append(int(i[3]))

        dataset.src = torch.tensor(src)
        dataset.dst = torch.tensor(dst)
        dataset.t = torch.tensor(t)
        dataset.msg = torch.vstack(msg)
        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        dataset.t = dataset.t.to(torch.long)
        if recording:
            torch.save(dataset, graphs_dir + "/graph_" + str(day) + ".TemporalData.simple")
        graphs.append(dataset)
    return graphs
