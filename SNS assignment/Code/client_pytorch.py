# client_pytorch.py
import socket

HOST = 'localhost'
PORT = 12345

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
query = input("Enter your query (e.g., 'Predict next day box office'): ")
client_socket.send(query.encode('utf-8'))
response = client_socket.recv(1024).decode('utf-8')
print("Server response:", response)
client_socket.close()
