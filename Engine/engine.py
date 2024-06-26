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

from .engine_config import *
from .engine_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
max_node_num = 268243  #! the number of nodes in node2id table +1
min_dst_idx, max_dst_idx = 0, max_node_num
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
abnormal_output = open("./anormaly.txt","w",encoding="utf-8")
dangerous_output = open("./dangerous.txt","w",encoding="utf-8")

# Some common path abstraction for visualization
def replace_path_name(path_name):
    for i in REPLACE_DICT:
        if i in path_name:
            return REPLACE_DICT[i]
    return path_name
 
# Users should manually put the detected anomalous time windows here

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
    loss=[]
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)

def gen_feature(cur):
    # Firstly obtain all node labels
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Construct the hierarchical representation for each node label
    node_msg_dic_list = []
    for i in tqdm(nodeid2msg.keys()):
        if type(i) == int:
            higlist = None
            #* nodeid2msg[i].keys() 即 node_type
            if 'netflow' in nodeid2msg[i].keys():
                higlist = ['netflow']
                higlist += ip2higlist(nodeid2msg[i]['netflow'])

            if 'file' in nodeid2msg[i].keys():
                higlist = ['file']
                higlist += path2higlist(nodeid2msg[i]['file'])

            if 'subject' in nodeid2msg[i].keys():
                higlist = ['subject']
                higlist += path2higlist(nodeid2msg[i]['subject'])
            if higlist is not None:
                node_msg_dic_list.append(list2str(higlist))

    # Featurize the hierarchical node labels
    FH_string = FeatureHasher(n_features=NODE_EMBEDDING_DIM, input_type="string")
    node2higvec=[]
    for i in tqdm(node_msg_dic_list):
        vec=FH_string.transform([i]).toarray()
        node2higvec.append(vec)
    node2higvec = np.array(node2higvec).reshape([-1, NODE_EMBEDDING_DIM])
    #! File Saved
    torch.save(node2higvec, ARTIFACT_DIR + "node2higvec")
    return node2higvec


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
            torch.cat(
                [
                    torch.from_numpy(node2higvec[i[0]]), 
                    rel2vec[i[2]], 
                    torch.from_numpy(node2higvec[i[1]])
                    ]
            )
        )
        t.append(int(i[3]))
    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    print("BreakPoint")
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
    #! File Saved
    #! torch.save(node_IDF, ARTIFACT_DIR + "node_IDF")
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

def analyse(graph,memory,gnn,link_pred,neighbor_loader,nodeid2msg):
    print("Preparing")
    test(
        inference_data=graph,
        memory=memory,
        gnn=gnn,
        link_pred=link_pred,
        neighbor_loader=neighbor_loader,
        nodeid2msg=nodeid2msg,
        path=ARTIFACT_DIR + "graph_list"
    )

def load_data():
    '''
    initialize and load prepared data
    '''
    print("[*] Loading Data")
    cur, _ = init_database_connection()
    #* rel2vec = gen_relation_onehot()
    #* node2higvec = gen_feature(cur=cur)
    #* rel2vec = torch.load(ARTIFACT_DIR + "rel2vec")
    #* node2higvec = torch.load(ARTIFACT_DIR + "node2higvec")
    #* events = listen()
    
    #? 会提示 RuntimeError: vstack expects a non-empty TensorList
    #* graph = gen_vectorized_graphs(events, node2higvec=node2higvec, rel2vec=rel2vec)
    print("[*] build graph")
    graph = torch.load("./artifact/graph_4_7.TemporalData.simple").to(device=device)
    nodeid2msg = gen_nodeid2msg(cur=cur)
    print("[*] load model")
    memory, gnn, link_pred, neighbor_loader = torch.load(f"{MODELS_DIR}/models.pt",map_location=device)
    return graph,memory,gnn,link_pred,neighbor_loader,nodeid2msg

def attack_investigate():
    print("Investigating Attacks")
    original_edges_count = 0
    gg = nx.DiGraph()
    count = 0
    for path in tqdm(ATTACK_LIST):
        if ".txt" in path:
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
        attack_nodes = ATTACK_NODES[DETECTION_LEVEL]
        flag = False
        for i in attack_nodes:
            if i in msg:
                flag = True
                break
        return flag

    # Plot and render candidate subgraph
    os.system(f"mkdir -p {ARTIFACT_DIR}/graph_visual/")

    graph_index = 0

    for c in communities:
        dot = Digraph(name="MyPicture", comment="the test", format="pdf")
        dot.graph_attr['rankdir'] = 'LR'

        for e in communities[c].edges:
            try:
                temp_edge = gg.edges[e]
            except:
                pass

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
                dangerous_output.write(f"{temp_edge['srcmsg']} --{temp_edge['edge_type']}--{temp_edge['dstmsg']}/{temp_edge['time']}\n")
                edge_color = 'red'
            else:
                abnormal_output.write(f"{temp_edge['srcmsg']} --{temp_edge['edge_type']}--{temp_edge['dstmsg']}/{ns_time_to_datetime(temp_edge['time'])}\n")
                edge_color = 'blue'
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                    str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=temp_edge['edge_type'],
                    color=edge_color)

        dot.render(f'{ARTIFACT_DIR}/graph_visual/subgraph_' + str(graph_index), view=False)
        graph_index += 1

def aberration_investigate():
    print("Investigating Aberration")
    node_IDF, tw_list = compute_IDF()
    anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{ARTIFACT_DIR}/graph_list/"
    )

def main():
    graph,memory,gnn,link_pred,neighbor_loader,nodeid2msg = load_data()
    analyse(graph,memory,gnn,link_pred,neighbor_loader,nodeid2msg)
    aberration_investigate()
    attack_investigate()

if __name__ == "__main__":
    main()
