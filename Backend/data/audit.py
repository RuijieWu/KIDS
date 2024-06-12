import os
import subprocess

def add_socket_audit_rule():
    """
    Add an audit rule using auditctl command.
    """
    try:
        subprocess.run(['sudo', 'auditctl', '-a', 'always,exit', '-F', 'arch=b64', '-S', 'sendto', '-S', 'recvfrom', '-k', 'socket_operations'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to add audit rule: {e}")

def add_file_watch_rule(directory):
    """
    Add a file watch rule for a specific directory.
    """
    try:
        subprocess.run(['auditctl', '-w', directory, '-p', 'rwxa', '-k', 'folder_watch'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to add file watch rule: {e}")

def restart_auditd_service():
    """
    Restart the auditd service.
    """
    try:
        subprocess.run(['systemctl', 'restart', 'auditd'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to restart auditd service: {e}")

def main():
    folder_to_watch = '/home/ubuntu/softbei/KIDS'  # Update with your folder path
    
    # Check if the script is run as root
    if os.geteuid() != 0:
        print("This script must be run as root.")
        return
    
    # Restart auditd service
    restart_auditd_service()

    # Add audit rules
    add_file_watch_rule(folder_to_watch)
    add_socket_audit_rule()


    print("Audit rules added and auditd service restarted successfully.")

if __name__ == "__main__":
    main()
