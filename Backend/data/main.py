import os
import subprocess
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class DirectoryPaths(BaseModel):
    paths: List[str]

# Function to format time for ausearch
def format_time(time):
    return time.strftime('%m/%d/%Y %H:%M:%S')

# Function to run ausearch
def run_ausearch(keyword, start_time, end_time):
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

# Function to extract timestamp in nanoseconds
def extract_timestamp_nsec(line):
    if 'msg=audit(' in line:
        timestamp_str = line.split('msg=audit(')[1].split(')')[0]
        seconds_str, nano_str = timestamp_str.split(':')
        timestamp_dt = datetime.utcfromtimestamp(float(seconds_str))
        timestamp_nsec = int(float(seconds_str) * 1_000_000_000) + int(nano_str)
        return timestamp_nsec
    return None

# Function to parse folder watch logs
def parse_folder_watch_logs(logs):
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

# Function to parse socket operations logs
def parse_socket_operations_logs(logs):
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

# Function to add socket audit rule
def add_socket_audit_rule():
    try:
        subprocess.run(['sudo', 'auditctl', '-a', 'always,exit', '-F', 'arch=b64', '-S', 'sendto', '-S', 'recvfrom', '-k', 'socket_operations'], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to add audit rule: {e}")

# Function to add file watch rule
def add_file_watch_rule(directory):
    try:
        subprocess.run(['auditctl', '-w', directory, '-p', 'rwxa', '-k', 'folder_watch'], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to add file watch rule: {e}")

# Function to restart auditd service
def restart_auditd_service():
    try:
        subprocess.run(['systemctl', 'restart', 'auditd'], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart auditd service: {e}")

# Function to clear audit logs
def clear_audit_logs():
    try:
        subprocess.run(['sudo', 'rm', '-rf', '/var/log/audit'], check=True)
        os.makedirs('/var/log/audit', exist_ok=True)  # Ensure the directory exists
        subprocess.run(['sudo', 'touch', '/var/log/audit/audit.log'], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear audit logs: {e}")

@app.post("/setup-audit")
def setup_audit(directories: DirectoryPaths):
    folder_paths = directories.paths  # Get the folder paths from the request body
    
    # Check if the script is run as root
    if os.geteuid() != 0:
        raise HTTPException(status_code=403, detail="This script must be run as root.")
    
    # Clear audit logs
    clear_audit_logs()
    
    # Restart auditd service
    restart_auditd_service()

    # Add audit rules for each directory
    for folder in folder_paths:
        add_file_watch_rule(folder)
    add_socket_audit_rule()

    return {"message": "Audit logs cleared, audit rules added, and auditd service restarted successfully."}

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
