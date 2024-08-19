import psycopg2
import time

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

# 执行SELECT ALL操作
def select_all(cursor, table_name):
    select_query = f"SELECT * FROM {table_name} WHERE time >= 1523011275000000000 AND time <= 1523099181000000000"1
    try:
        start_time = time.time()
        cursor.execute(select_query)
        cursor.fetchall()  # 获取所有结果
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to SELECT ALL from {table_name}: {elapsed_time:.4f} seconds")
    except Exception as error:
        print(f"Error selecting data from {table_name}: {error}")

def main():
    table_names = [
        "anomalous_actions_table",
        "anomalous_objects_table",
        "anomalous_subjects_table",
        "dangerous_actions_table",
        "dangerous_objects_table",
        "dangerous_subjects_table"
    ]

    connection = connect_to_db(db_config)
    if connection:
        cursor = connection.cursor()
        for table_name in table_names:
            select_all(cursor, table_name)
        cursor.close()
        connection.close()

if __name__ == "__main__":
    main()
