import os
import sys
import time
import torch
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm
from graphviz import Digraph
import networkx as nx
import community.community_louvain as community_louvain
from config import *

from model import *
from config import *
from utils import *
from embedder import *
from investigator import *

#* {
#*     Time: time_interval
#*     Loss: loss:.4f
#*     Nodes_count: len(unique_nodes)
#*     Edges_count: event_count
#* }
#* eg.
#* 2024-06-08 22:33:42 - INFO - Time: 2018-04-07 23:03:54.806921896~2018-04-07 23:20:00.066899749, Loss: 1.1485, Nodes_count: 56599, Edges_count: 92160
#* 2024-06-08 22:33:47 - INFO - Time: 2018-04-07 23:20:00.066899749~2018-04-07 23:35:19.036879610, Loss: 0.6923, Nodes_count: 56608, Edges_count: 14336
#* 2024-06-08 22:33:51 - INFO - Time: 2018-04-07 23:35:19.036879610~2018-04-07 23:50:19.096860042, Loss: 0.7274, Nodes_count: 56908, Edges_count: 14336


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
    os.system(f"mkdir -p {path}")

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
            edge_type = REL2ID[DETECTION_LEVEL][edgeindex]
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

def load_data(cur,begin_time,end_time):
    '''
    initialize and load data
    '''
    print("[*] Loading Data")

    print("[*] Loading nodeid2msg")
    nodeid2msg = gen_nodeid2msg(cur=cur)

    print("[*] Loading rel2vec")
    rel2vec = gen_relation_onehot()
    #* rel2vec = torch.load(ARTIFACT_DIR + "rel2vec")

    print("[*] Loading node2higvec")
    node2higvec = gen_feature(nodeid2msg)
    #* node2higvec = torch.load(ARTIFACT_DIR + "node2higvec")

    print("[*] Loading graphs")
    graphs = gen_vectorized_graphs(
        cur=cur,
        node2higvec=node2higvec,
        rel2vec=rel2vec,
        begin_time=begin_time,
        end_time=end_time
    )
    #* graph = torch.load("./artifact/graph_7.TemporalData.simple").to(device=device)

    print("[*] Loading model")
    memory, gnn, link_pred, neighbor_loader = torch.load(MODELS_PATH,map_location=device)
    #* memory, gnn, link_pred, neighbor_loader = torch.load(f"{MODELS_DIR}/models.pt",map_location=device)

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

def attack_investigate(cur,connect):
    print("Investigating Attacks")
    gg, communities, partition = community_discover()
    # Plot and render candidate subgraph
    os.system(f"mkdir -p {ARTIFACT_DIR}/graph_visual/")
    graph_index = 0
    dangerous_subjects = []
    dangerous_objects = []
    dangerous_actions = []
    anomalous_subjects = []
    anomalous_objects = []
    anomalous_actions = []

    for c in communities:
        dot = Digraph(name="IntrusionDetectionGraph", comment="KIDS Engine Output", format="png")
        dot.graph_attr['rankdir'] = 'LR'
        begin_time = BEGIN_TIME
        end_time = END_TIME
        for e in communities[c].edges:
            try:
                temp_edge = gg.edges[e]
                #* srcnode = e['srcnode']
                #* dstnode = e['dstnode']
            except Exception as _:
                return
            if temp_edge['time'] > end_time:
                end_time = temp_edge['time']
            if temp_edge['time'] < begin_time:
                begin_time = temp_edge['time']
            subject_node_name = temp_edge['srcmsg'][13:-2]
            if "'subject': '" in temp_edge['srcmsg']:
                src_shape = 'box'
                subject_node_type = "Subject"
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape = 'oval'
                subject_node_type = "File"
                subject_node_name = temp_edge['srcmsg'][10:-2]
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape = 'diamond'
                subject_node_type = "Netflow"
            else:
                src_shape = DEFAULT_SHAPE
                subject_node_type = "Netflow"
            if attack_edge_flag(temp_edge['srcmsg']):
                src_node_color = 'red'
                dangerous_subjects.append([temp_edge['time'],subject_node_type,subject_node_name])
            else:
                src_node_color = 'blue'
                anomalous_subjects.append([temp_edge['time'],subject_node_type,subject_node_name])
            dot.node(name=str(hashgen(replace_path_name(temp_edge['srcmsg']))), label=str(
                replace_path_name(temp_edge['srcmsg']) + str(
                    partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])), color=src_node_color,
                    shape=src_shape)

                # destination node
            object_node_name = temp_edge['dstmsg'][13:-2]
            if "'subject': '" in temp_edge['dstmsg']:
                dst_shape = 'box'
                object_node_type = "Subject"
            elif "'file': '" in temp_edge['dstmsg']:
                dst_shape = 'oval'
                object_node_type = "File"
                object_node_name = temp_edge['dstmsg'][10:-2]
            elif "'netflow': '" in temp_edge['dstmsg']:
                dst_shape = 'diamond'
                object_node_type = "Netflow"
            else:
                dst_shape = DEFAULT_SHAPE
                object_node_type = "Netflow"
            if attack_edge_flag(temp_edge['dstmsg']):
                dst_node_color = 'red'
                dangerous_objects.append([temp_edge['time'],object_node_type,object_node_name])
            else:
                dst_node_color = 'blue'
                anomalous_objects.append([temp_edge['time'],object_node_type,object_node_name])
            dot.node(name=str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=str(
                replace_path_name(temp_edge['dstmsg']) + str(
                    partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])), color=dst_node_color,
                    shape=dst_shape)

            if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']):
                edge_color = 'red'
                dangerous_actions.append([
                    temp_edge['time'],
                    subject_node_type,
                    subject_node_name,
                    temp_edge['edge_type'],
                    object_node_type,
                    object_node_name
                ])
            else:
                edge_color = 'blue'
                anomalous_actions.append([
                    temp_edge['time'],
                    subject_node_type,
                    subject_node_name,
                    temp_edge['edge_type'],
                    object_node_type,
                    object_node_name
                ])
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                    str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=temp_edge['edge_type'],
                    color=edge_color)

        #*dot.render(f"{ARTIFACT_DIR}/graph_visual/subgraph_{begin_time}_{end_time}_{str(graph_index)}", view=False)
        save_dangerous_actions(cur,connect,dangerous_actions)
        save_dangerous_subjects(cur,connect,dangerous_subjects)
        save_dangerous_objects(cur,connect,dangerous_objects)
        save_anomalous_actions(cur,connect,anomalous_actions)
        save_anomalous_subjects(cur,connect,anomalous_subjects)
        save_anomalous_objects(cur,connect,anomalous_objects)
        begin_time = ns_time_to_datetime(begin_time)
        end_time = ns_time_to_datetime(end_time)
        dot.render(f"{ARTIFACT_DIR}/graph_visual/{begin_time}~{end_time}_{str(graph_index)}", view=False)
        graph_index += 1

def aberration_investigate(cur,connect,recoding = False):
    print("Investigating Aberration")
    node_IDF, tw_list = compute_IDF()
    history_list = anomalous_queue_construction(
        cur=cur,
        connect=connect,
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{ARTIFACT_DIR}/graph_list/"
    )
    if recoding:
        torch.save(history_list, f"{ARTIFACT_DIR}/graph_history_list")

def arg_parse(args: list[str]):
    try:
        print(args[0])
        if args[1] in ("--help","-h"):
            print(HELP_MSG)
        elif args[1] == "init":
            dataset_path = "./"
            dataset = "CADETS-E3"
            for index,arg in enumerate(args):
                if arg.lower() in ("-path","--path"):
                    if args[index+1][-1] == "/":
                        dataset_path = args[index+1]
                    else:
                        dataset_path = args[index+1][:-1]
                if arg.lower() in ("-dataset","--dataset"):
                    if args[index+1] in ("CADETS-E3","CADETS-E5","CADETS-TC"):
                        dataset = args[index+1]
                    else:
                        return False, "error", "[*] Dataset Format provided is not allowed"
            return "init",dataset_path,dataset
        elif args[1] == "run":
            begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            for index,arg in enumerate(args):
                if arg.lower() in ("-begin","--begin"):
                    begin_time = f"{args[index+1]} 00:00:00"
                    try:
                        if ":" in args[index+2]:
                            begin_time = f"{args[index+1]} {args[index+2]}"
                    except IndexError:
                        pass
                if arg.lower() in ("-end","--end"):
                    end_time = f"{args[index+1]} 00:00:00"
                    try:
                        if ":" in args[index+2]:
                            end_time = f"{args[index+1]} {args[index+2]}"
                    except IndexError:
                        pass
            return "run",begin_time,end_time
        else:
            return False, "error", "[*] Wrong Arguments!"
    except IndexError:
        return False, "error", "[*] Wrong Arguments!"

def init():
    cur , connect = init_database_connection()
    cur.execute(DROP_TABLES)
    connect.commit()
    cur.execute(CREATE_PLUGIN)
    connect.commit()
    cur.execute(CREATE_EVENT_TABLE)
    connect.commit()
    cur.execute(CREATE_FILE_NODE_TABLE)
    connect.commit()
    cur.execute(CREATE_NETFLOW_NODE_TABLE)
    connect.commit()
    cur.execute(CREATE_SUBJECT_NODE_TABLE)
    connect.commit()
    cur.execute(CREATE_NODE2ID)
    connect.commit()
    cur.execute(CREATE_ABERRATION_STATICS_TABLE)
    connect.commit()
    cur.execute(CREATE_SUBJECTS_TABLE)
    connect.commit()
    cur.execute(CREATE_ACTIONS_TABLE)
    connect.commit()
    cur.execute(CREATE_OBJECTS_TABLE)
    connect.commit()

def main():
    '''
    Entrance of this Engine
    '''
    arguments = arg_parse(sys.argv)
    if not arguments[0]:
        print(arguments[2])
        sys.exit(0)
    if arguments[0] == "init":
        init()
    if arguments[0] == "run":
        begin_time = arguments[1]
        end_time = arguments[2]
        cur, connect = init_database_connection()
    #    graph,memory,gnn,link_pred,neighbor_loader,nodeid2msg = load_data(
    #        cur=cur,
    #        begin_time=begin_time,
    #        end_time=end_time
    #    )
    #    analyse(
    #        graphs=graph,
    #        memory=memory,
    #        gnn=gnn,
    #        link_pred=link_pred,
    #        neighbor_loader=neighbor_loader,
    #        nodeid2msg=nodeid2msg
    #    )
        aberration_investigate(
            cur=cur,
            connect=connect
        )
        attack_investigate(
            cur=cur,
            connect=connect
        )

if __name__ == "__main__":
    main()
