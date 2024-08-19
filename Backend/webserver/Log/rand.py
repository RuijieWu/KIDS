import psycopg2
import os
import random
import string

# 数据库连接配置
db_config = {
    'dbname': 'tc_cadet_dataset_db',
    'user': 'postgres',
    'password': 'postgres',
    'host': '/var/run/postgresql/',
    'port': '5432'
}

# 连接到数据库
conn = psycopg2.connect(**db_config)
cur = conn.cursor()

# 创建表（如果不存在）
cur.execute('''
    CREATE TABLE IF NOT EXISTS random_data (
        id SERIAL PRIMARY KEY,
        data TEXT
    )
''')
conn.commit()

# 生成随机字符串数据
def generate_random_data(size_in_bytes):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(size_in_bytes))

# 插入2GB随机数据
target_size_in_bytes = 2 * 1024 * 1024 * 1024  # 2GB
batch_size_in_bytes = 1024 * 1024  # 1MB per batch

total_inserted = 0
while total_inserted < target_size_in_bytes:
    random_data = generate_random_data(batch_size_in_bytes)
    cur.execute("INSERT INTO random_data (data) VALUES (%s)", (random_data,))
    total_inserted += batch_size_in_bytes
    conn.commit()
    print(f"Inserted {total_inserted / (1024 * 1024)} MB")

# 关闭连接
cur.close()
conn.close()
