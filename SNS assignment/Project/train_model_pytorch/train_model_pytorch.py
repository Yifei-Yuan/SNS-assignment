# train_model_pytorch.py
# train_model_pytorch.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 1. Load and preprocess data from CSV
data_file = 'Jurassic World Movie Database.csv'
df = pd.read_csv(data_file)  # 修改为读取 CSV 格式

# Convert 'Date' to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Create 'Day_Since_Release'
df['Day_Since_Release'] = (df['Date'] - df['Date'].min()).dt.days

# Log-transform the 'Daily' revenue (using log1p to handle zeros)
df['Log_Daily'] = np.log1p(df['Daily'])

# Select relevant columns (we'll use 'Log_Daily' and 'Theaters' as features)
processed_df = df[['Day_Since_Release', 'Daily', 'Log_Daily', 'Theaters', 'Rank']]

# Extract features and convert to numpy array
data = processed_df[['Log_Daily', 'Theaters']].values

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# 2. Construct sequence dataset using a sliding window approach
sequence_length = 7  # Use 7 days to predict the 8th day
X, y = [], []
for i in range(len(normalized_data) - sequence_length):
    X.append(normalized_data[i:i + sequence_length])
    y.append(normalized_data[i + sequence_length][0])  # Target: Log_Daily

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# 3. Create PyTorch Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# 4. Define the LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)  # out shape: (batch_size, sequence_length, hidden_size)
        out = out[:, -1, :]  # Use the output from the last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = 2
hidden_size = 64
num_layers = 1
output_size = 1
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            val_loss += loss.item() * batch_X.size(0)
    val_loss /= len(test_dataset)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 6. Evaluate the model on the test set
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(batch_y.tolist())

predictions = np.array(predictions).reshape(-1, 1)
actuals = np.array(actuals).reshape(-1, 1)


# Function to inverse transform predictions: reverse normalization and log transform
def inverse_transform_prediction(log_values):
    # Create a temporary array with an extra column for 'Theaters' (dummy zeros)
    temp = np.hstack((log_values, np.zeros((len(log_values), 1))))
    inv = scaler.inverse_transform(temp)[:, 0]
    return np.expm1(inv)  # Reverse log1p


y_pred_actual = inverse_transform_prediction(predictions)
y_test_actual = inverse_transform_prediction(actuals)

plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='Actual')
plt.plot(y_pred_actual, label='Predicted')
plt.title('Jurassic World Daily Box Office Prediction (PyTorch)')
plt.xlabel('Time (Days)')
plt.ylabel('Daily Box Office ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Save the trained model
torch.save(model.state_dict(), 'pytorch_model.pth')
print("Model saved as pytorch_model.pth")
