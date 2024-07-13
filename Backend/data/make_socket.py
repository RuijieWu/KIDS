import socket
import threading
import time

# 处理客户端连接
def handle_client(client_socket):
    while True:
        try:
            data = client_socket.recv(1024)
            if not data:
                break
            client_socket.sendall(data)  # 回显数据
        except ConnectionResetError:
            break
    client_socket.close()

# 启动服务器
def start_server(host, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Server listening on {host}:{port}")
    
    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

# 客户端线程
def client_thread(host, port, message, num_messages):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        for _ in range(num_messages):
            client.sendall(message.encode())
            response = client.recv(1024)
            print(f"Received: {response.decode()}")
            time.sleep(0.1)  # 等待一会儿再发送下一个消息
        client.close()
    except ConnectionRefusedError:
        print("Connection failed. Make sure the server is running.")

# 启动多个客户端
def start_clients(host, port, message, num_clients, num_messages):
    threads = []
    for i in range(num_clients):
        t = threading.Thread(target=client_thread, args=(host, port, message, num_messages))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

if __name__ == "__main__":
    # 配置服务器和客户端参数
    server_host = "127.0.0.1"
    server_port = 9999
    client_message = "Hello, Server!"
    num_clients = 10
    num_messages_per_client = 5
    
    # 启动服务器线程
    server_thread = threading.Thread(target=start_server, args=(server_host, server_port))
    server_thread.daemon = True
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(1)
    
    # 启动客户端
    start_clients(server_host, server_port, client_message, num_clients, num_messages_per_client)
    
    # 等待所有客户端完成
    server_thread.join()