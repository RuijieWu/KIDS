import logging
import os
import time
import torch
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm
from .engine_config import *
from .engine_utils import *

def log_events():
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

max_node_num = 268243  # the number of nodes in node2id table +1
min_dst_idx, max_dst_idx = 0, max_node_num
# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
    loss=[]
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)

def gen_vectorized_graphs(events, node2higvec, rel2vec):
    '''
    遍历数据库中的事件数据，根据指定条件选择需要的事件。
    将事件转换为图数据的形式，包括源节点、目标节点、消息（节点特征和关系编码的拼接）、时间戳等信息。
    将构建的图数据集保存在文件中
    '''
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
        #! File Saved
        #! torch.save(dataset, GRAPHS_DIR + "/graph_4_" + str(day) + ".TemporalData.simple")
    return dataset

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

    # Record the running time to evaluate the performance
    start = time.perf_counter()

    for batch in inference_data.seq_batches(batch_size=BATCH):

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
            time_with_loss[time_interval] = {'loss': loss,

                                             'nodes_count': len(unique_nodes),
                                             'total_edges': total_edges,
                                             'costed_time': (end - start)}

            log = open(path + "/" + time_interval + ".txt", 'w')

            for e in edge_list:
                loss += e['loss']

            loss = loss / event_count
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

#! To DIY
def load_data():
    # graph_4_3 - graph_4_5 will be used to initialize node IDF scores.
    graph_4_3 = torch.load(GRAPHS_DIR + "/graph_4_3.TemporalData.simple").to(device=device)
    graph_4_4 = torch.load(GRAPHS_DIR + "/graph_4_4.TemporalData.simple").to(device=device)
    graph_4_5 = torch.load(GRAPHS_DIR + "/graph_4_5.TemporalData.simple").to(device=device)

    # Testing set
    graph_4_6 = torch.load(GRAPHS_DIR + "/graph_4_6.TemporalData.simple").to(device=device)
    graph_4_7 = torch.load(GRAPHS_DIR + "/graph_4_7.TemporalData.simple").to(device=device)

    return [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7]

def gen_relation_onehot():
    relvec=torch.nn.functional.one_hot(
        torch.arange(0, len(REL2ID.keys())//2), 
        num_classes=len(REL2ID.keys())//2
    )
    rel2vec={}
    for i in REL2ID.keys():
        if type(i) is not int:
            rel2vec[i]= relvec[REL2ID[i]-1]
            rel2vec[relvec[REL2ID[i]-1]]=i
    torch.save(rel2vec, ARTIFACT_DIR + "rel2vec")
    return rel2vec

def listen():
    events = [[]]
    return events

def compute_IDF():
    '''
    根据给定的文件列表，计算结点 IDF ， 并在将计算结果保存至 IDF 文件中后返回计算结果与给定的文件列表
    '''
    node_IDF = {}

    file_list = []
    file_path = ARTIFACT_DIR + "graphs"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    node_set = {}
    for f_path in tqdm(file_list):
        f = open(f_path)
        for line in f:
            l = line.strip()
            jdata = eval(l)
            if jdata['loss'] > 0:
                if 'netflow' not in str(jdata['srcmsg']):
                    if str(jdata['srcmsg']) not in node_set.keys():
                        node_set[str(jdata['srcmsg'])] = {f_path}
                    else:
                        node_set[str(jdata['srcmsg'])].add(f_path)
                if 'netflow' not in str(jdata['dstmsg']):
                    if str(jdata['dstmsg']) not in node_set.keys():
                        node_set[str(jdata['dstmsg'])] = {f_path}
                    else:
                        node_set[str(jdata['dstmsg'])].add(f_path)
    for n in node_set:
        include_count = len(node_set[n])
        IDF = math.log(len(file_list) / (include_count + 1))
        node_IDF[n] = IDF

    torch.save(node_IDF, ARTIFACT_DIR + "node_IDF")
    return node_IDF, file_list

# Measure the relationship between two time windows, if the returned value
# is not 0, it means there are suspicious nodes in both time windows.
#! 打桩
def cal_set_rel(s1, s2, node_IDF, tw_list):
    '''
    计算时间窗口之间的关系，检查是否存在可疑结点
    '''
    def is_include_key_word(s):
        # The following common nodes don't exist in the training/validation data, but
        # will have the influences to the construction of anomalous queue (i.e. noise).
        # These nodes frequently exist in the testing data but don't contribute much to
        # the detection (including temporary files or files with random name).
        # Assume the IDF can keep being updated with the new time windows, these
        # common nodes can be filtered out.
        keywords = KEYWORDS[DETECTION_LEVEL]
        flag = False
        for i in keywords:
            if i in s:
                flag = True
        return flag

    new_s = s1 & s2
    count = 0
    for i in new_s:
        if is_include_key_word(i) is True:
            node_IDF[i] = math.log(len(tw_list) / (1 + len(tw_list)))

        if i in node_IDF.keys():
            IDF = node_IDF[i]
        else:
            # Assign a high IDF for those nodes which are neither in training/validation
            # sets nor excluded node list above.
            IDF = math.log(len(tw_list) / (1))

        # Compare the IDF with a rareness threshold α
        if IDF > (math.log(len(tw_list) * 0.9)):
            count += 1
    return count
def cal_anomaly_loss(loss_list, edge_list):
    '''
    接收 损失列表，边列表 为参数
    1. 对比损失列表与边列表的值找出其中值异常的结点和边
    2. 计算出异常损失的数量，平均损失
    返回上述工作流程中得到的结果
    '''
    if len(loss_list) != len(edge_list):
        print("error!")
        return 0
    count = 0
    loss_sum = 0
    loss_std = std(loss_list)
    loss_mean = mean(loss_list)
    edge_set = set()
    node_set = set()

    thr = loss_mean + 1.5 * loss_std


    for i in range(len(loss_list)):
        if loss_list[i] > thr:
            count += 1
            src_node = edge_list[i][0]
            dst_node = edge_list[i][1]
            loss_sum += loss_list[i]

            node_set.add(src_node)
            node_set.add(dst_node)
            edge_set.add(edge_list[i][0] + edge_list[i][1])
    return count, loss_sum / count, node_set, edge_set


#! 打桩
def anomalous_queue_construction(node_IDF, tw_list, graph_dir_path):
    '''
    执行异常队列的构建，遍历给定目录中的文件，计算每个时间窗口内的异常情况，并根据一定逻辑增量地构建队列
    '''
    history_list = []
    current_tw = {}

    file_l = os.listdir(graph_dir_path)
    index_count = 0
    for f_path in sorted(file_l):

        f = open(f"{graph_dir_path}/{f_path}")
        edge_loss_list = []
        edge_list = []

        # Figure out which nodes are anomalous in this time window
        for line in f:
            l = line.strip()
            jdata = eval(l)
            edge_loss_list.append(jdata['loss'])
            edge_list.append([str(jdata['srcmsg']), str(jdata['dstmsg'])])
        count, loss_avg, node_set, edge_set = cal_anomaly_loss(edge_loss_list, edge_list)
        current_tw['name'] = f_path
        current_tw['loss'] = loss_avg
        current_tw['index'] = index_count
        current_tw['nodeset'] = node_set

        # Incrementally construct the queues
        added_que_flag = False
        for hq in history_list:
            for his_tw in hq:
                if cal_set_rel(current_tw['nodeset'], his_tw['nodeset'], node_IDF, tw_list) != 0 and current_tw['name'] != his_tw['name']:
                    hq.append(copy.deepcopy(current_tw))
                    added_que_flag = True
                    break
                if added_que_flag:
                    break
        if added_que_flag is False:
            temp_hq = [copy.deepcopy(current_tw)]
            history_list.append(temp_hq)

        index_count += 1

    return history_list



def main():
    rel2vec = gen_relation_onehot()
    node2higvec = torch.load(ARTIFACT_DIR + "node2higvec").to(device=device)
    events = listen()
    graph = gen_vectorized_graphs(events, node2higvec=node2higvec, rel2vec=rel2vec)
    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur=cur)
    memory, gnn, link_pred, neighbor_loader = torch.load(f"{MODELS_DIR}/models.pt",map_location=device)
    test(
        inference_data=graph,
        memory=memory,
        gnn=gnn,
        link_pred=link_pred,
        neighbor_loader=neighbor_loader,
        nodeid2msg=nodeid2msg,
        path=ARTIFACT_DIR + "graph"
    )
    node_IDF, tw_list = compute_IDF()
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{ARTIFACT_DIR}/graph_4_6/"
    )
    torch.save(history_list, f"{ARTIFACT_DIR}/graph_4_6_history_list")
