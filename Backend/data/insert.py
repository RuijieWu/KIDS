import os
import json
from datetime import datetime, timedelta

def format_time(time):
    """
    Format a datetime object to the string format required by ausearch.
    """
    return time.strftime('%m/%d/%Y %H:%M:%S')

def run_ausearch(keyword, start_time, end_time):
    """
    Run ausearch with the specified keyword and time range, then return the output.
    """
    start_str = format_time(start_time)
    end_str = format_time(end_time)
    command = f"ausearch -k {keyword} -ts {start_str} -te {end_str}"
    print(f"Executing command: {command}")
    try:
        result = os.popen(command).read()
        return result
    except Exception as e:
        print(f"Failed to run ausearch: {e}")
        return None

def parse_folder_watch_logs(logs):
    """
    Parse ausearch logs specifically for folder_watch to extract the process name, file name, and event type.
    """
    events = []
    log_entries = logs.split('----\n')

    for entry in log_entries:
        lines = entry.split('\n')
        event = {}
        for line in lines:
            if line.startswith('type=SYSCALL'):
                if 'comm=' in line:
                    event['process'] = line.split('comm=')[1].split()[0].strip('"')
                if 'syscall=' in line:
                    syscall = int(line.split('syscall=')[1].split()[0])
                    if syscall == 257:
                        event['event'] = "EVENT_OPEN"
                    elif syscall == 2:
                        event['event'] = "EVENT_EXECUTE"
                    elif syscall == 5:
                        event['event'] = "EVENT_CLOSE"
                    elif syscall == 0:
                        event['event'] = "EVENT_READ"
                    elif syscall == 1:
                        event['event'] = "EVENT_WRITE"
            elif line.startswith('type=PATH'):
                if 'name=' in line:
                    event['file'] = line.split('name=')[1].split()[0].strip('"')
        if event:
            events.append(event)

    return events

def parse_socket_operations_logs(logs):
    """
    Parse ausearch logs specifically for socket_operations to extract the process name, addresses, and event type.
    """
    events = []
    log_entries = logs.split('----\n')

    for entry in log_entries:
        lines = entry.split('\n')
        event = {}
        src_ip = None
        src_port = None
        dest_ip = None
        dest_port = None
        for line in lines:
            if line.startswith('type=SYSCALL'):
                if 'comm=' in line:
                    event['process'] = line.split('comm=')[1].split()[0].strip('"')
                if 'syscall=' in line:
                    syscall = int(line.split('syscall=')[1].split()[0])
                    if syscall in [44, 45, 46]:  # syscalls for sendto, recvfrom, etc.
                        event['event'] = "EVENT_SENDTO" if syscall == 44 else "EVENT_RECVFROM"
            elif line.startswith('type=SOCKADDR'):
                if 'saddr=' in line:
                    saddr = line.split('saddr=')[1].strip()
                    if len(saddr) > 24:  # AF_INET or AF_INET6
                        src_ip = '.'.join(str(int(saddr[i:i+2], 16)) for i in range(0, 8, 2))
                        src_port = int(saddr[8:12], 16)
                        dest_ip = '.'.join(str(int(saddr[i:i+2], 16)) for i in range(12, 20, 2))
                        dest_port = int(saddr[20:24], 16)
        if event and 'process' in event and 'event' in event and src_ip and src_port and dest_ip and dest_port:
            event['source_ip'] = src_ip
            event['source_port'] = src_port
            event['destination_ip'] = dest_ip
            event['destination_port'] = dest_port
            events.append(event)

    return events

def write_to_file(filename, content):
    """
    Write content to a file.
    """
    with open(filename, 'w') as f:
        f.write(content)

def main():
    # Keywords used in the audit rules
    keywords = ['socket_operations', 'folder_watch']

    # Calculate the time range for the audit logs (past one hour)
    end_time = datetime.now()  # Current time
    start_time = end_time - timedelta(days=1)

    # Check audit logs for each keyword within the past one hour
    for keyword in keywords:
        logs = run_ausearch(keyword, start_time, end_time)
        if logs:
            if keyword == 'folder_watch':
                events = parse_folder_watch_logs(logs)
                filename = f"{keyword}_audit_logs.json"
                write_to_file(filename, json.dumps(events, indent=4))
                print(f"Audit logs for keyword '{keyword}' have been written to {filename}.")
            elif keyword == 'socket_operations':
                events = parse_socket_operations_logs(logs)
                filename = f"{keyword}_audit_logs.json"
                write_to_file(filename, json.dumps(events, indent=4))
                print(f"Audit logs for keyword '{keyword}' have been written to {filename}.")
        else:
            print(f"No logs found for keyword '{keyword}'.")

if __name__ == "__main__":
    main()
