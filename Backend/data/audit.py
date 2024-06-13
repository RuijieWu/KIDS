import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class DirectoryPath(BaseModel):
    path: str

def add_socket_audit_rule():
    """
    Add an audit rule using auditctl command.
    """
    try:
        subprocess.run(['sudo', 'auditctl', '-a', 'always,exit', '-F', 'arch=b64', '-S', 'sendto', '-S', 'recvfrom', '-k', 'socket_operations'], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to add audit rule: {e}")

def add_file_watch_rule(directory):
    """
    Add a file watch rule for a specific directory.
    """
    try:
        subprocess.run(['auditctl', '-w', directory, '-p', 'rwxa', '-k', 'folder_watch'], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to add file watch rule: {e}")

def restart_auditd_service():
    """
    Restart the auditd service.
    """
    try:
        subprocess.run(['systemctl', 'restart', 'auditd'], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart auditd service: {e}")

def clear_audit_logs():
    """
    Remove existing audit logs and create a new empty log file.
    """
    try:
        subprocess.run(['sudo', 'rm', '-rf', '/var/log/audit'], check=True)
        os.makedirs('/var/log/audit', exist_ok=True)  # Ensure the directory exists
        subprocess.run(['sudo', 'touch', '/var/log/audit/audit.log'], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear audit logs: {e}")

@app.post("/setup-audit")
def setup_audit(directory: DirectoryPath):
    folder_to_watch = directory.path  # Get the folder path from the request body
    
    # Check if the script is run as root
    if os.geteuid() != 0:
        raise HTTPException(status_code=403, detail="This script must be run as root.")
    
    # Clear audit logs
    clear_audit_logs()
    
    # Restart auditd service
    restart_auditd_service()

    # Add audit rules
    add_file_watch_rule(folder_to_watch)
    add_socket_audit_rule()

    return {"message": "Audit logs cleared, audit rules added, and auditd service restarted successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
