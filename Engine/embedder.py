'''
Embed Events from database into GNN
'''

from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm

import numpy as np
import torch

from config import *
from utils import *

def gen_feature(nodeid2msg, rendering = False):
    '''
    gen_feature
    '''
    node_msg_dic_list = []
    for i in tqdm(nodeid2msg.keys(),desc="Generating Feature"):
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

    FH_string = FeatureHasher(n_features=NODE_EMBEDDING_DIM, input_type="string")
    node2higvec=[]
    for i in tqdm(node_msg_dic_list,desc="Generating Feature"):
        vec=FH_string.transform([i]).toarray()
        node2higvec.append(vec)
    node2higvec = np.array(node2higvec).reshape([-1, NODE_EMBEDDING_DIM])
    if rendering:
        torch.save(node2higvec, ARTIFACT_DIR + "node2higvec")
    return node2higvec

def gen_relation_onehot(rendering = False):
    '''
    gen_relation_onehot
    '''
    relvec = torch.nn.functional.one_hot(
        torch.arange(0, len(REL2ID[DETECTION_LEVEL].keys())//2),
        num_classes=len(REL2ID[DETECTION_LEVEL].keys())//2
    )
    rel2vec = {}
    for i in REL2ID[DETECTION_LEVEL].keys():
        if type(i) is not int:
            rel2vec[i]= relvec[REL2ID[DETECTION_LEVEL][i]-1]
            rel2vec[relvec[REL2ID[DETECTION_LEVEL][i]-1]]=i
    if rendering:
        torch.save(rel2vec, ARTIFACT_DIR + "rel2vec")
    return rel2vec

def gen_vectorized_graphs(
    cur,
    node2higvec,
    rel2vec,
    begin_time,
    end_time,
    rendering = False
):
    '''
    gen_vectorized_graphs
    '''
    graphs = []
    for interval_time in tqdm(range(begin_time,end_time,TIME_INTERVAL),desc="Generating Vectorized Graphs"):
        begin_timestamp = interval_time
        end_timestamp = interval_time + TIME_INTERVAL
        end_timestamp = end_time if end_time < end_timestamp else end_timestamp
        events = get_events(cur,begin_timestamp,end_timestamp)
        if not events:
            continue
        edge_list = []
        for e in events:
            edge_temp = [int(e[1]), int(e[4]), e[2], e[5]]
            if e[2] in EDGE_TYPE[DETECTION_LEVEL]:
                edge_list.append(edge_temp)

        dataset = TemporalData()
        src = []
        dst = []
        msg = []
        t = []
        for i in edge_list:
            src.append(int(i[0]))
            dst.append(int(i[1]))
            msg.append(torch.cat([
                    torch.from_numpy(node2higvec[i[0]]),
                    rel2vec[i[2]],
                    torch.from_numpy(node2higvec[i[1]])
                ]))
            t.append(int(i[3]))

        dataset.src = torch.tensor(src)
        dataset.dst = torch.tensor(dst)
        dataset.t = torch.tensor(t)
        dataset.msg = torch.vstack(msg)
        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        dataset.t = dataset.t.to(torch.long)

        if rendering:
            torch.save(
                dataset,
               GRAPHS_DIR + "/graph_" + str(begin_time) + '~' + str(end_time) + ".TemporalData.simple"
            )
        graphs.append(dataset)
    return graphs
