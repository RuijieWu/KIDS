'''
Show Database Output
'''

from tqdm import tqdm
from utils import *
from config import *

def get_aberration_statics(cur, rendering):
    '''
    get_aberration_statics
    '''
    sql = """
    select * from aberration_statics_table;
    """
    cur.execute(sql)
    results = cur.fetchall()
    if rendering:
        logger = open(LOG_DIR + "aberration_statics.csv","a",encoding="utf-8")
    for result in tqdm(results,desc="get_aberration_statics"):
        if result[3] > MIN_AVG_LOSS and result[3] < MAX_AVG_LOSS:
            result = (f"{result[0]}",*result[1:])
            if rendering:
                logger.write(f"{result}\n")
            print(
      #*f"{ns_time_to_datetime(result[0])[:-10]}~{ns_time_to_datetime(result[1])[:-10]} Average loss:{result[2]}"
        f"{ns_time_to_datetime_US(result[0])}~{ns_time_to_datetime_US(result[1])}.txt\n" +\
        f"Average loss:{result[2]}\n" +\
        f"Num of anomalous edges within the time window: {result[3]}\n"+\
#        f"Percentage of anomalous edges: {result[4]}\n" +\
#        f"Anomalous node count: {result[5]}\n"+\
#        f"Anomalous edge count: {result[6]}\n"+\
        f"Loss Sum: {result[2]*result[3]}"
            )

def get_anomalous_actions(cur, rendering):
    '''
    get_anomalous_actions
    '''
    sql = """
    select * from anomalous_actions_table;
    """
    cur.execute(sql)
    results = cur.fetchall()
    if rendering:
        logger = open(LOG_DIR + "anomalous_actions.csv","a",encoding="utf-8")
    for result in tqdm(results,desc="get_anomalous_actions"):
        result = (f"{result[0]}",*result[1:])
        if rendering:
            logger.write(f"{result}\n")
        print(result)

def get_anomalous_subjects(cur, rendering):
    '''
    get_anomalous_subjects
    '''
    sql = """
    select * from anomalous_subjects_table;
    """
    cur.execute(sql)
    results = cur.fetchall()
    if rendering:
        logger = open(LOG_DIR + "anomalous_subjects.csv","a",encoding="utf-8")
    for result in tqdm(results,desc="get_anomalous_subjects"):
        result = (f"{result[0]}",*result[1:])
        if rendering:
            logger.write(f"{result}\n")
        print(result)

def get_anomalous_objects(cur, rendering):
    '''
    get_anomalous_objects
    '''
    sql = """
    select * from anomalous_objects_table;
    """
    cur.execute(sql)
    results = cur.fetchall()
    if rendering:
        logger = open(LOG_DIR + "anomalous_objects.csv","a",encoding="utf-8")
    for result in tqdm(results,desc="get_anomalous_objects"):
        result = (f"{result[0]}",*result[1:])
        if rendering:
            logger.write(f"{result}\n")
        print(result)

def get_dangerous_actions(cur, rendering):
    '''
    get_dangerous_actions
    '''
    sql = """
    select * from dangerous_actions_table;
    """
    cur.execute(sql)
    results = cur.fetchall()
    if rendering:
        logger = open(LOG_DIR + "dangerous_actions.csv","a",encoding="utf-8")
    for result in tqdm(results,desc="get_dangerous_actions"):
        result = (f"{result[0]}",*result[1:])
        if rendering:
            logger.write(f"{result}\n")
        print(result)

def get_dangerous_subjects(cur, rendering):
    '''
    get_dangerous_subjects
    '''
    sql = """
    select * from dangerous_subjects_table;
    """
    cur.execute(sql)
    results = cur.fetchall()
    if rendering:
        logger = open(LOG_DIR + "dangerous_subjects.csv","a",encoding="utf-8")
    for result in tqdm(results,desc="get_dangerous_subjects"):
        result = (f"{result[0]}",*result[1:])
        if rendering:
            logger.write(f"{result}\n")
        print(result)

def get_dangerous_objects(cur, rendering):
    '''
    get_dangerous_objects
    '''
    sql = """
    select * from dangerous_objects_table;
    """
    cur.execute(sql)
    results = cur.fetchall()
    if rendering:
        logger = open(LOG_DIR + "dangerous_objects.csv","a",encoding="utf-8")
    for result in tqdm(results,desc="get_dangerous_objects"):
        result = (f"{result[0]}",*result[1:])
        if rendering:
            logger.write(f"{result}\n")
        print(result)

def test(rendering):
    '''test'''
    cur , _ = init_database_connection()
    get_aberration_statics(cur,rendering)
    get_anomalous_subjects(cur,rendering)
    get_anomalous_actions(cur,rendering)
    get_anomalous_objects(cur,rendering)
    get_dangerous_subjects(cur,rendering)
    get_dangerous_actions(cur,rendering)
    get_dangerous_objects(cur,rendering)

if __name__ == "__main__":
    test(True)
