'''
Analyse System Logs
'''

import os

import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from utils import *
from config import *
from model import *

def compute_IDF(rendering = False):
    '''
    compute_IDF
    '''
    node_IDF = {}
    file_list = []
    file_path = f"{ARTIFACT_DIR}/graph_list/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)
    node_set = {}
    for f_path in tqdm(file_list,desc="Computing IDS"):
        f = open(f_path,encoding="utf-8")
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
    for key , value in node_set.items():
        include_count = len(value)
        idf = math.log(len(file_list) / (include_count + 1))
        node_IDF[key] = idf
    if rendering:
        torch.save(node_IDF, ARTIFACT_DIR + "node_IDF")
    return node_IDF, file_list

def cal_anomaly_loss(loss_list, edge_list, recording = True):
    '''
    cal_anomaly_loss
    '''
    if recording:
        logger = open(LOG_DIR + "cal_anomaly_loss.txt","a",encoding="utf-8")
    if len(loss_list) != len(edge_list):
        print("error!")
        return 0
    count = 0
    loss_sum = 0
    loss_std = std(loss_list)
    loss_mean = mean(loss_list)
    edge_set = set()
    node_set = set()

    thr = loss_mean + LOSS_FACTOR * loss_std

    if recording:
        logger.write(f"thr:{thr}\n")

    for i , _ in enumerate(loss_list):
        if loss_list[i] > thr:
            count += 1
            src_node = edge_list[i][0]
            dst_node = edge_list[i][1]
            loss_sum += loss_list[i]

            node_set.add(src_node)
            node_set.add(dst_node)
            edge_set.add(edge_list[i][0] + edge_list[i][1])
            if recording:
                logger.write(f"src_node:{src_node}->dst_node:{dst_node}\n")
                logger.write(f"loss:{loss_list[i]}\n")

    return count, loss_sum / count, node_set, edge_set

def cal_set_rel(s1, s2, node_IDF, tw_list, recording = True):
    '''
    cal_set_rel
    '''
    if recording:
        logger = open(LOG_DIR + "cal_set_rel.txt","a",encoding="utf-8")

    def is_include_key_word(s):
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
            IDF = math.log(len(tw_list) / (1))

        if IDF > (math.log(len(tw_list) * 0.9)):
            if recording:
                logger.write(f"node:{i}, IDF:{IDF}\n")
            count += 1
    return count

def anomalous_queue_construction(
    cur,
    connect,
    node_IDF,
    tw_list,
    graph_dir_path,
    recording = True
):
    '''
    anomalous_queue_construction
    '''
    if recording:
        logger = open(LOG_DIR + "anomalous_queue_construction.txt","a",encoding="utf-8")
    history_list = []
    current_tw = {}

    file_l = os.listdir(graph_dir_path)
    index_count = 0
    for f_path in tqdm(sorted(file_l),desc="Constructing Anomalous Queue"):
        if recording:
            logger.write("**************************************************\n")
            logger.write(f"Time window: {f_path}\n")

        f = open(f"{graph_dir_path}/{f_path}",encoding="utf-8")
        edge_loss_list = []
        edge_list = []
        if recording:
            logger.write(f'Time window index: {index_count}')

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

        if recording:
            logger.write(f"Average loss: {loss_avg}\n")
            logger.write(f"Num of anomalous edges within the time window: {count}\n")
            logger.write(f"Percentage of anomalous edges: {count / edge_count}\n")
            logger.write(f"Anomalous node count: {node_num}\n")
            logger.write(f"Anomalous edge count: {edge_num}\n")
            logger.write("**************************************************\n")

        begin_time = datetime_to_ns_time_US(f_path[:29])
        end_time = datetime_to_ns_time_US(f_path[30:-4])
        aberration_statics = [[
            f"{ns_time_to_datetime(begin_time)}",
            f"{ns_time_to_datetime(end_time)}",
            begin_time,
            end_time,
            loss_avg,
            count,
            count / count,
            node_num,
            edge_num
        ]]
        save_aberration_statics(
            cur,
            connect,
            aberration_statics
        )

    return history_list

def classifier_evaluation(y_test, y_test_pred, recording = True):
    '''
    classifier_evaluation
    '''
    if recording:
        logger = open(LOG_DIR + "classifier_evaluation.txt","a",encoding="utf-8")
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
    if recording:
        logger.write(f'tn: {tn} fp: {fp}')
        logger.write(f'tp: {tp} fn: {fn}')
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)
    if recording:
        logger.write(f"precision: {precision}")
        logger.write(f"recall: {recall}")
        logger.write(f"fscore: {fscore}")
        logger.write(f"accuracy: {accuracy}")
        logger.write(f"auc_val: {auc_val}")
    return precision,recall,fscore,accuracy,auc_val

def ground_truth_label(attack_list):
    '''
    ground_truth_label
    '''
    labels = {}
    for f in os.listdir(f"{ARTIFACT_DIR}/graph_list"):
        labels[f] = 0

    for i in attack_list:
        labels[i] = 1

    return labels

def evaluate(history_list, attack_list, recording = True):
    '''
    evaluate
    '''
    if recording:
        logger = open(LOG_DIR + "evaluate.txt","a",encoding="utf-8")
    pred_label = {}
    for f in os.listdir(f"{ARTIFACT_DIR}/graph_list"):
        pred_label[f] = 0

    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > BETA_DAY:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            if recording:
                logger.write(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            if recording:
                logger.write(f"Anomaly score: {anomaly_score}")

    labels = ground_truth_label(attack_list)
    y = []
    y_pred = []
    for key , value in labels.items():
        y.append(value)
        y_pred.append(pred_label[key])
    classifier_evaluation(y, y_pred)

def calc_attack_edges(attack_list):
    '''
    calc_attack_edges
    '''
    def keyword_hit(line):
        attack_nodes = ATTACK_NODES[DETECTION_LEVEL]
        flag = False
        for i in attack_nodes:
            if i in line:
                flag = True
                break
        return flag

    files = []
    for f in attack_list:
        files.append(f"{ARTIFACT_DIR}/graph_list/{f}")

    attack_edge_count = 0
    for fpath in (files):
        f = open(fpath,encoding="utf-8")
        for line in f:
            if keyword_hit(line):
                attack_edge_count += 1

def analyse(cur,connect,begin_time,end_time,rendering = False):
    '''
    analyse
    '''
    print("[*] Analyzing")
    node_IDF, tw_list = compute_IDF()
    history_list = anomalous_queue_construction(
        cur=cur,
        connect=connect,
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{ARTIFACT_DIR}/graph_list/"
    )
    attack_list = get_attack_list(cur,begin_time,end_time)
    evaluate(history_list,attack_list)
    if rendering:
        torch.save(history_list, f"{ARTIFACT_DIR}/graph_history_list")
