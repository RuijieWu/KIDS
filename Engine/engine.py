'''
Main Program of KIDS Engine
'''

import os
import sys
import time
import torch
from copy import deepcopy
from flask import (
    Flask,
    jsonify
)
#import numpy as np
#from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
import torch.nn as nn
from tqdm import tqdm

from analyser import analyse
from model import cal_pos_edges_loss_multiclass
from config import (
    config,
    device,
    HELP_MSG,
    ALLOWED_CMD,
    FORBIDDEN_KEYS
)
from utils import (
    init_database_connection,
    gen_nodeid2msg,
    Command,
    datetime_to_ns_time_US,
    ns_time_to_datetime_US,
    tensor_find,
)
from embedder import (
    gen_relation_onehot,
    gen_feature,
    gen_vectorized_graphs
)
from investigator import investigate


@torch.no_grad()
def test(
    inference_data,
    memory,
    gnn,
    link_pred,
    neighbor_loader,
    nodeid2msg,
    path,
    recording = True
):
    '''
    测试函数，接收 推理数据，内存模型，图神经网络，链接预测器，邻居加载器，节点ID到消息的映射，路径 为参数
    1. 针对模型测试进行一系列的初始化
    2. 逐批次遍历推理数据，计算并记录损失值
    返回时间窗口内的损失情况
    '''
    os.system(f"mkdir -p {path}")
    if recording:
        logger = open(config["LOG_DIR"] + "test.txt","a",encoding="utf-8")

    criterion = nn.CrossEntropyLoss()
    max_node_num = 268243  #! the number of nodes in node2id table +1
    # min_dst_idx, max_dst_idx = 0, max_node_num
    assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

    memory.eval()
    gnn.eval()
    link_pred.eval()

    memory.reset_state()
    neighbor_loader.reset_state()

    time_with_loss = {}
    total_loss = 0
    edge_list = []

    unique_nodes = torch.tensor([]).to(device=device)
    total_edges = 0

    start_time = inference_data.t[0]
    event_count = 0
    pos_o = []

    start = time.perf_counter()

    for batch in inference_data.seq_batches(batch_size=config["BATCH"]):

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
        total_edges += config["BATCH"]

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
            l = tensor_find(m[config["NODE_EMBEDDING_DIM"]:-config["NODE_EMBEDDING_DIM"]], 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)
        total_loss += float(loss) * batch.num_events

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)

        for i in range(len(pos_out)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])

            srcmsg = str(nodeid2msg[srcnode])
            dstmsg = str(nodeid2msg[dstnode])
            t_var = int(t[i])
            edgeindex = tensor_find(msg[i][config["NODE_EMBEDDING_DIM"]:-config["NODE_EMBEDDING_DIM"]], 1)
            edge_type = config["REL2ID"][config["DETECTION_LEVEL"]][edgeindex]
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
        if t[-1] > start_time + int(config["TIME_WINDOW_SIZE"]):
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
            if recording:
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
    memory, gnn, link_pred, neighbor_loader = torch.load(config["ARTIFACT_DIR"] + "models/" + config["MODEL_NAME"] + ".pt",map_location=device)
    #* memory, gnn, link_pred, neighbor_loader = torch.load(f"{MODELS_DIR}/models.pt",map_location=device)

    return graphs, memory, gnn, link_pred, neighbor_loader, nodeid2msg

def test_data(
    graphs,
    memory,
    gnn,
    link_pred,
    neighbor_loader,
    nodeid2msg
):
    '''
    test_data
    '''
    #* print("[*] Testing Data")
    #* loss_list = []
    for graph in tqdm(graphs,desc="Testing Data"):
        loss = test(
            inference_data=graph,
            memory=memory,
            gnn=gnn,
            link_pred=link_pred,
            neighbor_loader=neighbor_loader,
            nodeid2msg=nodeid2msg,
            path=config["ARTIFACT_DIR"] + "graph_list"
        )
        #* loss_list.append(loss)
    #* return loss_list

def init():
    '''
    init
    '''
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

def api(command: Command):
    '''
    work as api server
    '''
    app = Flask(__name__)
    host = command.api_args.get("host",config["DEFAULT_HOST"])
    port = command.api_args.get("port",config["DEFAULT_PORT"])

    @app.route("/ping")
    def ping():
        return jsonify({
            "status": "200 OK",
            "msg": "pong!\n"
            }), 200

    @app.route("/api/<cmd>/<begin_time>/<end_time>")
    def kids_api(cmd:str, begin_time, end_time):
        cmd = cmd.lower()
        if cmd not in ALLOWED_CMD:
            return jsonify({
                "status": "400 Bad Request",
                "msg": f"To {cmd} is not allowed"
                }), 400
        run(begin_time,end_time,cmd)
        return jsonify({
            "status": "200 OK",
            "msg": f"{cmd} successfully"
            }), 200

    @app.route("/config/update/<key>/<value>")
    def update(key:str, value):
        key = key.upper()
        value = value.replace('|','/')
        if key not in config.keys() or \
            key in FORBIDDEN_KEYS:
            return jsonify({
                "status": "400 Bad Request",
                "msg": f"Key {key} is not allowed"
                }), 400
        config[key] = value
        return jsonify({
            "status": "200 OK",
            "msg": f"{key} has been updated to {value}"
            }), 200

    @app.route("/config/view")
    def view():
        data = deepcopy(config)
        for key in FORBIDDEN_KEYS:
            data.pop(key)
        return jsonify({
            "status":"200 OK",
            "config":data
            }), 200

    app.run(
        host=host,
        port=port
    )

def run(begin_time, end_time, cmd):
    '''
    run KIDS engine
    '''
    if len(begin_time) < 20:
        begin_time = begin_time + ".000000000"
    if len(end_time) < 20:
        end_time = end_time + ".000000000"
    begin_timestamp = datetime_to_ns_time_US(begin_time)
    end_timestamp = datetime_to_ns_time_US(end_time)
    cur, connect = init_database_connection()
    if cmd in ("run","test"):
        graphs,memory,gnn,link_pred,neighbor_loader,nodeid2msg = load_data(
            cur=cur,
            begin_time=begin_timestamp,
            end_time=end_timestamp
        )
        #* loss_list = test_data()
        test_data(
            graphs=graphs,
            memory=memory,
            gnn=gnn,
            link_pred=link_pred,
            neighbor_loader=neighbor_loader,
            nodeid2msg=nodeid2msg
        )
    if cmd in ("run","analyse"):
        analyse(
            cur=cur,
            connect=connect,
            begin_time=begin_timestamp,
            end_time=end_timestamp
        )
    if cmd in ("run","investigate"):
        investigate(
            cur=cur,
            connect=connect,
            begin_time=begin_timestamp,
            end_time=end_timestamp
        )

def main():
    '''
    Entrance of this Engine
    '''
    try:
        command = Command()
        command.parse(sys.argv)
        if command.help or not command.cmd:
            print(HELP_MSG)
            sys.exit(0)
        cmd = command.cmd
        if cmd in ("init"):
            init()
        elif cmd in ALLOWED_CMD:
            begin_time = command.begin_time
            end_time = command.end_time
            cmd = command.cmd
            run(begin_time,end_time,cmd)
        elif cmd in ("rpc", "flask", "api"):
            api(command)
        else:
            print("[*] Wrong Arguments. Try --help for more information")
    except KeyboardInterrupt:
        print("[*] Engine Terminated")
        sys.exit(0)

if __name__ == "__main__":
    main()
