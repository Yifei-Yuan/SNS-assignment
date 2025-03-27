# server_pytorch.py
import socket
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from train_model_pytorch import LSTMModel  # 确保 LSTMModel 定义与训练时一致

# 定义模型参数
input_size = 2
hidden_size = 64
num_layers = 1
output_size = 1

# 初始化模型并加载训练好的权重
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('pytorch_model.pth'))
model.eval()

# 重新构建 scaler：从 CSV 文件加载原始数据，并建立归一化对象
data_file = 'Jurassic World Movie Database.csv'
df = pd.read_csv(data_file)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Log_Daily'] = np.log1p(df['Daily'])
df_processed = df[['Log_Daily', 'Theaters']].values
scaler = MinMaxScaler()
scaler.fit(df_processed)

def prepare_input():
    # 使用数据中最后 7 天作为输入
    normalized_data = scaler.transform(df_processed)
    input_seq = normalized_data[-7:]
    input_seq = np.expand_dims(input_seq, axis=0)  # 形状: (1, 7, 2)
    return input_seq

def inverse_transform_prediction(pred):
    # 将预测结果从归一化及对数转换后还原到原始票房值
    temp = np.hstack((pred, np.zeros((len(pred), 1))))
    inv = scaler.inverse_transform(temp)[:, 0]
    return np.expm1(inv)

HOST = 'localhost'
PORT = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)


print("Server running on {}:{}".format(HOST, PORT))

while True:
    client_socket, addr = server_socket.accept()
    print("Connected by", addr)
    request = client_socket.recv(1024).decode('utf-8')
    print("Received request:", request)
    input_seq = prepare_input()
    input_seq_tensor = torch.tensor(input_seq, dtype=torch.float32)
    with torch.no_grad():
        pred = model(input_seq_tensor)
    pred_np = pred.detach().numpy()
    pred_actual = inverse_transform_prediction(pred_np)
    response = "Predicted next day box office: ${:.2f}".format(pred_actual[0])
    client_socket.send(response.encode('utf-8'))
    client_socket.close()
