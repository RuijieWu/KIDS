'''
Date: 2024-06-12 22:36:44
LastEditTime: 2024-07-13 20:10:48
Description: 
'''
import os
from graphviz import Digraph
import networkx as nx
import community.community_louvain as community_louvain
from tqdm import tqdm

from utils import *
from config import *
from model import *

def community_discover(attack_list):
    '''
    community_discover
    '''
    original_edges_count = 0
    #* graphs = []
    gg = nx.DiGraph()
    count = 0
    for path in tqdm(attack_list):
        if ".txt" in path:
            #* line_count = 0
            #* node_set = set()
            tempg = nx.DiGraph()
            f = open(f"{ARTIFACT_DIR}/graph_list/{path}", "r",encoding="utf-8")
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
            thr = loss_mean + LOSS_FACTOR * loss_std
            for e in edge_list:
                if e['loss'] > thr:
                    tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),
                                str(hashgen(replace_path_name(e['dstmsg']))))
                    gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))), str(hashgen(replace_path_name(e['dstmsg']))),
                                loss=e['loss'], srcmsg=e['srcmsg'], dstmsg=e['dstmsg'], edge_type=e['edge_type'],
                                time=e['time'])


    partition = community_louvain.best_partition(gg.to_undirected())

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
    '''
    attack_edge_flag
    '''
    attack_nodes = ATTACK_NODES[DETECTION_LEVEL]
    flag = False
    for i in attack_nodes:
        if i in msg:
            flag = True
    return flag

def investigate(cur,connect,begin_time,end_time):
    '''
    investigate
    '''
    print("[*] Investigating")
    attack_list = get_attack_list(cur,begin_time,end_time)
    gg, communities, partition = community_discover(attack_list)
    os.system(f"mkdir -p {ARTIFACT_DIR}/graph_visual/")
    #* graph_index = 0
    dangerous_subjects = []
    dangerous_objects = []
    dangerous_actions = []
    anomalous_subjects = []
    anomalous_objects = []
    anomalous_actions = []

    for c in tqdm(communities):
        graph_index = 0
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
            dot.node(
                name=str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                label=str(
                    replace_path_name(temp_edge['srcmsg']) + \
                    str(partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])
                ),
                color=src_node_color,
                shape=src_shape
            )

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
            dot.node(
                name=str(hashgen(replace_path_name(temp_edge['dstmsg']))),
                label=str(
                    replace_path_name(temp_edge['dstmsg']) + \
                    str(partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])
                ),
                color=dst_node_color,
                shape=dst_shape
            )

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
            dot.edge(
                str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                str(hashgen(replace_path_name(temp_edge['dstmsg']))),
                label=temp_edge['edge_type'][6:],
                color=edge_color
            )

        #*dot.render(
        #*f"{ARTIFACT_DIR}/graph_visual/subgraph_{begin_time}_{end_time}_{str(graph_index)}",
        #*view=False)
        save_dangerous_actions(cur,connect,dangerous_actions)
        save_dangerous_subjects(cur,connect,dangerous_subjects)
        save_dangerous_objects(cur,connect,dangerous_objects)
        save_anomalous_actions(cur,connect,anomalous_actions)
        save_anomalous_subjects(cur,connect,anomalous_subjects)
        save_anomalous_objects(cur,connect,anomalous_objects)
        begin_time = ns_time_to_datetime_US(begin_time)
        end_time = ns_time_to_datetime_US(end_time)
        dot.render(
            filename=f"{ARTIFACT_DIR}/graph_visual/{begin_time}~{end_time}_{str(graph_index)}",
            view=False
        )
        graph_index += 1
