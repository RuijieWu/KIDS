import os
import psycopg2


# Function to insert data into PostgreSQL from CSV
def insert_data_from_csv(filename):
    table_name = os.path.splitext(os.path.basename(filename))[0]+"_table"
    with open(filename, 'r') as f:
        for line in f:
            if line:
                command = f"INSERT INTO {table_name} VALUES ({line.strip()});"
                # write command to a txt
                with open('insert_commands.txt', 'a') as file:
                    file.write(command + "\n")

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
