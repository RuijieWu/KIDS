'''
Date: 2024-06-12 22:36:44
LastEditTime: 2024-06-14 22:50:10
Description: 
'''
import torch
import os
from graphviz import Digraph
import networkx as nx
import community.community_louvain as community_louvain
from tqdm import tqdm

from config import *
from utils import *

#? Aberration Investigate
#* {
#*     TimeWindow: 2018-04-05 00:00:00.019021794~2018-04-05 00:15:00.828996051.txt
#*     TimeWindowIndex: 0
#*     Thr: thr // 异常的阈值
#*     Count: count // 超过阈值的边的数量
#*     Nodes_count: len(unique_nodes)
#*     Edges_count: event_count
#* }
#* .eg
#* 2024-06-09 01:19:49 - INFO - Time window: 2018-04-05 00:00:00.019021794~2018-04-05 00:15:00.828996051.txt
#* 2024-06-09 01:19:49 - INFO - Time window index: 0
#* 2024-06-09 01:19:50 - INFO - thr:1.9376321934097978
#* 2024-06-09 01:19:50 - INFO - Average loss: 2.422255516815709
#* 2024-06-09 01:19:50 - INFO - Num of anomalous edges within the time window: 1093
#* 2024-06-09 01:19:50 - INFO - Percentage of anomalous edges: 0.08894856770833333
#* 2024-06-09 01:19:50 - INFO - Anomalous node count: 73
#* 2024-06-09 01:19:50 - INFO - Anomalous edge count: 87

#? Attack Investigation
#* Subject:{SubjectType,SubjectName}
#* Action:{ActionType}
#* Object:{ObjectType,ObjectName}
#* .eg
#*	9380699345889908536 -> 199462899167661811 [label=EVENT_WRITE color=blue]
#*	9380699345889908536 [label="{'subject': 'syslogd'}9" color=blue shape=box]
#*	6635167680731577424 [label="{'file': '/var/log/auth.log'}9" color=blue shape=oval]
#*	9380699345889908536 -> 6635167680731577424 [label=EVENT_WRITE color=blue]

def cal_anomaly_loss(loss_list, edge_list):
    logger = open("./aberration_investigation.txt","a",encoding="utf-8")
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

    logger.write(f"thr:{thr}\n")

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

def compute_IDF(recording = False):
    node_IDF = {}
    file_list = []
    file_path = f"{ARTIFACT_DIR}/graph_list/"
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
    if recording:
        torch.save(node_IDF, ARTIFACT_DIR + "node_IDF")
    return node_IDF, file_list

# Measure the relationship between two time windows, if the returned value
# is not 0, it means there are suspicious nodes in both time windows.
def cal_set_rel(s1, s2, node_IDF, tw_list):
    logger = open("./aberration_investigation.txt","a",encoding="utf-8")
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
            logger.write(f"node:{i}, IDF:{IDF}\n")
            count += 1
    return count

def anomalous_queue_construction(
    cur,
    connect,
    node_IDF,
    tw_list,
    graph_dir_path
):
    logger = open("./aberration_investigation.txt","a",encoding="utf-8")
    history_list = []
    current_tw = {}

    file_l = os.listdir(graph_dir_path)
    index_count = 0
    for f_path in sorted(file_l):
        logger.write("**************************************************\n")
        logger.write(f"Time window: {f_path}\n")

        f = open(f"{graph_dir_path}/{f_path}")
        edge_loss_list = []
        edge_list = []
        logger.write(f'Time window index: {index_count}')

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
        edge_count = len(edge_list)
        node_num = len(node_set)
        edge_num = len(edge_set)
        logger.write(f"Average loss: {loss_avg}")
        logger.write(f"Num of anomalous edges within the time window: {count}")
        logger.write(f"Percentage of anomalous edges: {count / edge_count}")
        logger.write(f"Anomalous node count: {node_num}")
        logger.write(f"Anomalous edge count: {edge_num}")
        logger.write("**************************************************\n")
        save_aberration_statics(
            cur,
            connect,
            f_path[:-4],
            loss_avg,
            count,
            count / count,
            node_num,
            edge_num
        )

    return history_list

def community_discover():
    original_edges_count = 0
    #* graphs = []
    gg = nx.DiGraph()
    count = 0
    attack_list = ATTACK_LIST[DETECTION_LEVEL]
    if not attack_list:
        for file in os.listdir(f"{ARTIFACT_DIR}/graph_list"):
            attack_list.append(f"{ARTIFACT_DIR}/graph_list/{file}")
    for path in tqdm(attack_list):
        if ".txt" in path:
            #* line_count = 0
            #* node_set = set()
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

    return gg, communities, partition

def attack_edge_flag(msg):
    attack_nodes = ATTACK_NODES[DETECTION_LEVEL]
    flag = False
    for i in attack_nodes:
        if i in msg:
            flag = True
    return flag
