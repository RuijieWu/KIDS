import os
import psycopg2

# Define PostgreSQL connection parameters
conn_params = {
    'host': '/var/run/postgresql/',
    'user': 'postgres',
    'password': 'postgres',
    'dbname': 'tc_cadet_dataset_db',
    'port': '5432',
    'sslmode': 'disable',
}

# Function to print table structure
def print_table_structure(table_name):
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()

        # Query to fetch table structure
        cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'")
        columns = cur.fetchall()

        # Print table structure
        print(f"Table Structure for {table_name}:")
        for column in columns:
            print(f"{column[0]} - {column[1]}")

        cur.close()
    except psycopg2.Error as e:
        print(f"Error fetching table structure: {e}")
    finally:
        if conn:
            conn.close()

# Function to insert data into PostgreSQL from CSV
def insert_data_from_csv(filename):
    table_name = os.path.splitext(os.path.basename(filename))[0]+"_table"
    print_table_structure(table_name)  # Print table structure

    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()

        with open(filename, 'r') as f:
            next(f)  # Skip header line
            for line in f:
                if line:
                    try:
                        command = f"INSERT INTO {table_name} VALUES ({line.strip()})"
                        print(command)
                        cur.execute(command)
                    except psycopg2.Error as e:
                        print(f"Error inserting row into {table_name}: {e}")

        # After insertion, fetch and print first 5 rows to verify
        cur.execute(f"SELECT * FROM {table_name} LIMIT 5")
        rows = cur.fetchall()
        print(f"First 5 rows inserted into {table_name}:")
        for row in rows:
            print(row)

    except psycopg2.Error as e:
        print(f"Error inserting data into {table_name}: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# Main execution
if __name__ == "__main__":
    csv_files = [
        'aberration_statics.csv',
        'anomalous_actions.csv',
        'anomalous_objects.csv',
        'anomalous_subjects.csv',
        'dangerous_objects.csv',
        'dangerous_subjects.csv'
    ]

    for csv_file in csv_files:
        insert_data_from_csv(csv_file)
