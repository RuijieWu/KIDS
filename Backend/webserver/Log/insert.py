import psycopg2

# 数据库连接配置
db_config = {
    'dbname': 'tc_cadet_dataset_db',
    'user': 'postgres',
    'password': 'postgres',
    'host': '/var/run/postgresql/',
    'port': '5432'
}

# 连接到数据库
def connect_to_db(config):
    try:
        connection = psycopg2.connect(**config)
        return connection
    except Exception as error:
        print(f"Error connecting to database: {error}")
        return None

# 插入数据到数据库
def insert_data_0(cursor, data):
    insert_query = "INSERT INTO anomalous_actions_table(timestamp,time,subject_type,subject_name,action,object_type,object_name,graph_index) VALUES" + data
    try:
        # print query
        cursor.execute(insert_query)
        return True
    except Exception as error:
        print(f"Error inserting data: {error}")
        return False

def insert_data_1(cursor, data):
    insert_query = "INSERT INTO anomalous_objects_table(timestamp,time,object_type,object_name,graph_index) VALUES" + data
    try:
        # print query
        cursor.execute(insert_query)
        return True
    except Exception as error:
        print(f"Error inserting data: {error}")
        return False

def insert_data_2(cursor, data):
    insert_query = "INSERT INTO anomalous_subjects_table(timestamp,time,subject_type,subject_name,graph_index) VALUES" + data
    try:
        # print query
        cursor.execute(insert_query)
        return True
    except Exception as error:
        print(f"Error inserting data: {error}")
        return False

def insert_data_3(cursor, data):
    insert_query = "INSERT INTO dangerous_actions_table(timestamp,time,subject_type,subject_name,action,object_type,object_name,graph_index) VALUES" + data
    try:
        # print query
        cursor.execute(insert_query)
        return True
    except Exception as error:
        print(f"Error inserting data: {error}")
        return False

def insert_data_4(cursor, data):
    insert_query = "INSERT INTO dangerous_objects_table(timestamp,time,object_type,object_name,graph_index) VALUES" + data
    try:
        # print query
        cursor.execute(insert_query)
        return True
    except Exception as error:
        print(f"Error inserting data: {error}")
        return False

def insert_data_5(cursor, data):
    insert_query = "INSERT INTO dangerous_subjects_table(timestamp,time,subject_type,subject_name,graph_index) VALUES" + data
    try:
        # print query
        cursor.execute(insert_query)
        return True
    except Exception as error:
        print(f"Error inserting data: {error}")
        return False

def main():
    connection = connect_to_db(db_config)
    if connection:
        cursor = connection.cursor()
        
        # 清空相关数据库表
        tables = [
            "anomalous_actions_table",
            "anomalous_objects_table",
            "anomalous_subjects_table",
            "dangerous_actions_table",
            "dangerous_objects_table",
            "dangerous_subjects_table"
        ]
        
        try:
            for table in tables:
                cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
            connection.commit()
        except Exception as error:
            print(f"Error truncating tables: {error}")
            connection.rollback()
            cursor.close()
            connection.close()
            return
        with open("./anomalous_actions.csv", 'r') as csv_file:
            for line in csv_file:
                if insert_data_0(cursor, line) is False:
                    break
        connection.commit()
        with open("./anomalous_objects.csv", 'r') as csv_file:
            for line in csv_file:
                if insert_data_1(cursor, line) is False:
                    break
        connection.commit()
        with open("./anomalous_subjects.csv", 'r') as csv_file:
            for line in csv_file:
                if insert_data_2(cursor, line) is False:
                    break
        connection.commit()
        with open("./dangerous_actions.csv", 'r') as csv_file:
            for line in csv_file:
                if insert_data_3(cursor, line) is False:
                    break
        connection.commit()
        with open("./dangerous_objects.csv", 'r') as csv_file:
            for line in csv_file:
                if insert_data_4(cursor, line) is False:
                    break
        connection.commit()
        with open("./dangerous_subjects.csv", 'r') as csv_file:
            for line in csv_file:
                if insert_data_5(cursor, line) is False:
                    break
        connection.commit()
        cursor.close()
        connection.close()

if __name__ == "__main__":
    main()
