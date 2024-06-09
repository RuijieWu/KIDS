'''
在通过正常数据集训练出的模型通过测试集计算出测试集中所有数据的损失后，
该模块会从测试结果中筛选出损失值异常的数据，并为它们构建异常队列以供进一步的异常情况分析
'''
import logging
import torch
from kairos_utils import *
from config import *

# Setting for logging
logger = logging.getLogger("anomalous_queue_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ARTIFACT_DIR + 'anomalous_queue.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
#! 打桩
#! output = open("./jerry/output.txt","w",encoding="utf-8")

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

    logger.info(f"thr:{thr}")

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

#! To DIY
#* 将以下 file_path 转移至 config.py 中实现 or 自动进行遍历
def compute_IDF():
    '''
    根据给定的文件列表，计算结点 IDF ， 并在将计算结果保存至 IDF 文件中后返回计算结果与给定的文件列表
    '''
    node_IDF = {}

    file_list = []
    file_path = ARTIFACT_DIR + "graph_4_3/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    file_path = ARTIFACT_DIR + "graph_4_4/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    file_path = ARTIFACT_DIR + "graph_4_5/"
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
    logger.info("IDF weight calculate complete!")
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
            logger.info(f"node:{i}, IDF:{IDF}")
            count += 1
    return count

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
        logger.info("**************************************************")
        logger.info(f"Time window: {f_path}")

        f = open(f"{graph_dir_path}/{f_path}")
        edge_loss_list = []
        edge_list = []
        logger.info(f'Time window index: {index_count}')

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


        logger.info(f"Average loss: {loss_avg}")
        logger.info(f"Num of anomalous edges within the time window: {count}")
        logger.info(f"Percentage of anomalous edges: {count / len(edge_list)}")
        logger.info(f"Anomalous node count: {len(node_set)}")
        logger.info(f"Anomalous edge count: {len(edge_set)}")
        logger.info("**************************************************")

    return history_list


if __name__ == "__main__":
    logger.info("Start logging.")

    node_IDF, tw_list = compute_IDF()

    # Validation date
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{ARTIFACT_DIR}/graph_4_5/"
    )
    torch.save(history_list, f"{ARTIFACT_DIR}/graph_4_5_history_list")

    # Testing date
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{ARTIFACT_DIR}/graph_4_6/"
    )
    torch.save(history_list, f"{ARTIFACT_DIR}/graph_4_6_history_list")

    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{ARTIFACT_DIR}/graph_4_7/"
    )
    torch.save(history_list, f"{ARTIFACT_DIR}/graph_4_7_history_list")
