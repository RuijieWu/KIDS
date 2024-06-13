import os
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from typing import Optional

app = FastAPI()

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

def extract_timestamp_nsec(line):
    """
    Extract the timestamp from a line and convert it to nanoseconds.
    """
    if 'msg=audit(' in line:
        timestamp_str = line.split('msg=audit(')[1].split(')')[0]
        seconds_str, nano_str = timestamp_str.split(':')
        timestamp_dt = datetime.utcfromtimestamp(float(seconds_str))
        timestamp_nsec = int(float(seconds_str) * 1_000_000_000) + int(nano_str)
        return timestamp_nsec
    return None

def parse_folder_watch_logs(logs):
    """
    Parse ausearch logs specifically for folder_watch to extract the process name, file name, event type, and timestamp.
    """
    events = []
    log_entries = logs.split('----\n')

    for entry in log_entries:
        lines = entry.split('\n')
        event = {}
        for line in lines:
            if line.startswith('type=SYSCALL'):
                timestamp_nsec = extract_timestamp_nsec(line)
                if timestamp_nsec:
                    event['timestamp_rec'] = timestamp_nsec
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
    Parse ausearch logs specifically for socket_operations to extract the process name, addresses, event type, and timestamp.
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
                timestamp_nsec = extract_timestamp_nsec(line)
                if timestamp_nsec:
                    event['timestamp_rec'] = timestamp_nsec
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

@app.get("/audit-logs")
def get_audit_logs(start_time: str, end_time: str):
    try:
        start_time_dt = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
        end_time_dt = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Incorrect time format. Use 'YYYY-MM-DDTHH:MM:SS' format.")

    keywords = ['socket_operations', 'folder_watch']
    results = {}

    for keyword in keywords:
        logs = run_ausearch(keyword, start_time_dt, end_time_dt)
        if logs:
            if keyword == 'folder_watch':
                events = parse_folder_watch_logs(logs)
                results['folder_watch'] = events
            elif keyword == 'socket_operations':
                events = parse_socket_operations_logs(logs)
                results['socket_operations'] = events
        else:
            results[keyword] = []

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
