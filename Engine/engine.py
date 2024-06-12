import os
import time
import torch
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm
from graphviz import Digraph
import networkx as nx
import community.community_louvain as community_louvain

from model import *
from config import *
from utils import *
from embedder import *
from investigator import *

#* device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#* criterion = nn.CrossEntropyLoss()
#* max_node_num = 268243  #! the number of nodes in node2id table +1
#* min_dst_idx, max_dst_idx = 0, max_node_num
#* assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
abnormal_output = open("./anormaly.txt","w",encoding="utf-8")
dangerous_output = open("./dangerous.txt","w",encoding="utf-8")
logger = open("./aberration_investigation.txt","a",encoding="utf-8")

@torch.no_grad()
def test(
    inference_data,
    memory,
    gnn,
    link_pred,
    neighbor_loader,
    nodeid2msg,
    path
):
    '''
    测试函数，接收 推理数据，内存模型，图神经网络，链接预测器，邻居加载器，节点ID到消息的映射，路径 为参数
    1. 针对模型测试进行一系列的初始化
    2. 逐批次遍历推理数据，计算并记录损失值
    返回时间窗口内的损失情况
    '''
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
    logger = open("./analysis.txt","a",encoding="utf-8")
    logger.write(f"[*] Analysis at {time.ctime()}\n")
    ###? Init
    print("[*] Analysis Initializing")
    memory.eval()
    gnn.eval()
    link_pred.eval()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    time_with_loss = {}  # key: time，  value： the losses
    total_loss = 0
    edge_list = []

    unique_nodes = torch.tensor([]).to(device=device)
    total_edges = 0

    start_time = inference_data.t[0]
    event_count = 0
    pos_o = []
    ###?

    # Record the running time to evaluate the performance
    print("[*] Analyse Begin")
    start = time.perf_counter()
    
    for batch in inference_data.seq_batches(batch_size=BATCH):

        #*    dataset.src = torch.tensor(src)
        #*    dataset.dst = torch.tensor(dst)
        #*    dataset.t = torch.tensor(t)
        #*    dataset.msg = torch.vstack(msg)
        #*    dataset.src = dataset.src.to(torch.long)
        #*    dataset.dst = dataset.dst.to(torch.long)
        #*    dataset.msg = dataset.msg.to(torch.float)
        #*    dataset.t = dataset.t.to(torch.long)

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
        total_edges += BATCH

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        pos_o.append(pos_out)
        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg:
            l = tensor_find(m[NODE_EMBEDDING_DIM:-NODE_EMBEDDING_DIM], 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)
        total_loss += float(loss) * batch.num_events

        # update the edges in the batch to the memory and neighbor_loader
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # compute the loss for each edge
        each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)
        
        for i in range(len(pos_out)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])

            srcmsg = str(nodeid2msg[srcnode])
            dstmsg = str(nodeid2msg[dstnode])
            t_var = int(t[i])
            edgeindex = tensor_find(msg[i][NODE_EMBEDDING_DIM:-NODE_EMBEDDING_DIM], 1)
            edge_type = REL2ID[edgeindex]
            loss = each_edge_loss[i]

            temp_dic = {}
            temp_dic['loss'] = float(loss)
            temp_dic['srcnode'] = srcnode
            temp_dic['dstnode'] = dstnode
            temp_dic['srcmsg'] = srcmsg
            temp_dic['dstmsg'] = dstmsg
            temp_dic['edge_type'] = edge_type
            temp_dic['time'] = t_var

            edge_list.append(temp_dic)

        event_count += len(batch.src)
        if t[-1] > start_time + TIME_WINDOW_SIZE:
            # Here is a checkpoint, which records all edge losses in the current time window
            time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])
            end = time.perf_counter()
            time_with_loss[time_interval] = {
                'loss': loss,
                'nodes_count': len(unique_nodes),
                'total_edges': total_edges,
                'costed_time': (end - start)
            }

            log = open(path + "/" + time_interval + ".txt", 'w')

            for e in edge_list:
                loss += e['loss']

            loss = loss / event_count
            logger.write(
                f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s\n'
            )
            edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)  # Rank the results based on edge losses
            for e in edge_list:
                log.write(str(e))
                log.write("\n")
            event_count = 0
            total_loss = 0
            start_time = t[-1]
            log.close()
            edge_list.clear()

    return time_with_loss

def listen():
    cur, _ = init_database_connection()
    events=[]
    #! range(2,14)
    for day in tqdm(range(7, 8)):
        start_timestamp = datetime_to_ns_time_US('2018-04-' + str(day) + ' 00:00:00')
        end_timestamp = datetime_to_ns_time_US('2018-04-' + str(day + 1) + ' 00:00:00')
        sql = """
        select * from event_table
        where
              timestamp_rec>'%s' and timestamp_rec<'%s'
               ORDER BY timestamp_rec;
        """ % (start_timestamp, end_timestamp)
        cur.execute(sql)
        events += cur.fetchall()
    return events

def load_data():
    '''
    initialize and load data
    '''
    #* Embedding
    print("[*] Loading Data")
    cur, _ = init_database_connection()
    print("[*] Loading rel2vec")
    rel2vec = gen_relation_onehot()
    print("[*] Loading node2higvec")
    node2higvec = gen_feature(cur=cur)
    #* rel2vec = torch.load(ARTIFACT_DIR + "rel2vec")
    #* node2higvec = torch.load(ARTIFACT_DIR + "node2higvec")
    print("[*] Loading Events")
    #* events = listen()
    print("[*] Loading graph")
    #? 会提示 RuntimeError: vstack expects a non-empty TensorList
    #* graph = gen_vectorized_graphs(events, node2higvec=node2higvec, rel2vec=rel2vec)
    #* graph = torch.load("./artifact/graph_4_7.TemporalData.simple").to(device=device)
    graphs = gen_vectorized_graphs(cur=cur, node2higvec=node2higvec, rel2vec=rel2vec)
    
    #* test
    print("[*] Loading nodeid2msg")
    nodeid2msg = gen_nodeid2msg(cur=cur)
    print("[*] Loading model")
    #* memory, gnn, link_pred, neighbor_loader = torch.load(f"{MODELS_DIR}/models.pt",map_location=device)
    memory, gnn, link_pred, neighbor_loader = torch.load(f"{MODELS_DIR}cadets3_models.pt",map_location=device)

    return graphs, memory, gnn, link_pred, neighbor_loader, nodeid2msg

def analyse(
    graphs,
    memory,
    gnn,
    link_pred,
    neighbor_loader,
    nodeid2msg
):
    print("[*] Analysing Data")
    for graph in graphs:
        loss = test(
            inference_data=graph,
            memory=memory,
            gnn=gnn,
            link_pred=link_pred,
            neighbor_loader=neighbor_loader,
            nodeid2msg=nodeid2msg,
            path=ARTIFACT_DIR + "graph_list"
        )

def attack_investigate():
    print("Investigating Attacks")
    gg, communities, partition = community_discover()
    # Plot and render candidate subgraph
    os.system(f"mkdir -p {ARTIFACT_DIR}/graph_visual/")
    graph_index = 0
    for c in communities:
        dot = Digraph(name="MyPicture", comment="the test", format="pdf")
        dot.graph_attr['rankdir'] = 'LR'
        for e in communities[c].edges:
            try:
                temp_edge = gg.edges[e]
                #* srcnode = e['srcnode']
                #* dstnode = e['dstnode']
            except Exception as _:
                return
            if "'subject': '" in temp_edge['srcmsg']:
                src_shape = 'box'
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape = 'oval'
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape = 'diamond'
            else:
                src_shape = DEFAULT_SHAPE
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
            else:
                dst_shape = DEFAULT_SHAPE
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

        dot.render(f'{ARTIFACT_DIR}/graph_visual/subgraph_' + str(graph_index), view=False)
        graph_index += 1

def aberration_investigate(recoding = False):
    print("Investigating Aberration")
    node_IDF, tw_list = compute_IDF()
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{ARTIFACT_DIR}/graph_list/"
    )
    if recoding:
        torch.save(history_list, f"{ARTIFACT_DIR}/graph_history_list")

def main():
    #* graph,memory,gnn,link_pred,neighbor_loader,nodeid2msg = load_data()
    #* analyse(graph,memory,gnn,link_pred,neighbor_loader,nodeid2msg)
    aberration_investigate()
    attack_investigate()

if __name__ == "__main__":
    main()
