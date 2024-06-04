import numpy as np
import torch
import torch.nn as nn
# from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import networkx as nx
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
import matplotlib.pyplot as plt

with open('Flagged Data/flagged_data.pkl', 'rb') as file:
    flagged_data = pickle.load(file)

# Calculate the split point based on 70%
split_point = int(len(flagged_data) * 0.7)

# Split the dictionary
training_data = dict(list(flagged_data.items())[:split_point])
test_data = dict(list(flagged_data.items())[split_point:])

'''
# sensor_data = flagged_data[7] # temporarily work with the first day of data

# Extract day numbers and nested dictionary
day_nums = list(flagged_data.keys())
sensor_data = flagged_data.values()

# Extract sensor IDs and corresponding time series data
sensor_ids = list(sensor_data.keys())
time_series_flags = list(sensor_data.values())

# Create a fully connected graph for simplicity
num_sensors = len(sensor_ids)
adjacency_matrix = np.ones((num_sensors, num_sensors)) - np.eye(num_sensors)  # Fully connected minus self-loops

# Create the graph
graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

# Convert to PyTorch Geometric data
data = from_networkx(graph)
data.x = torch.tensor(np.array(time_series_flags), dtype=torch.float)
'''

# Define the dimensions of your input data
num_sensors = 7
num_features_per_sensor = 288
num_epochs = 10  # Number of training epochs
learning_rate = 0.01

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features_per_sensor, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = nn.Linear(32, num_sensors)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def dtw_loss(output, time_series_flags):
    # Compute DTW distances between the output embeddings and the original sensor data
    dtw_total = 0
    for i in range(len(time_series_flags)):
        for j in range(i + 1, len(time_series_flags)):
            distance, _ = fastdtw(output[i].detach().numpy(), output[j].detach().numpy()) #REMOVED dist=euclidean
            original_distance, _ = fastdtw(time_series_flags[i], time_series_flags[j])
            dtw_total += (distance - original_distance) ** 2
    return dtw_total / (len(time_series_flags) * (len(time_series_flags) - 1) / 2)

# Training loop
model.train()
for epoch in range(num_epochs):  # Number of epochs
    for day, day_data in training_data.items():
        # Extract sensor IDs and corresponding time series data
        sensor_ids = list(day_data.keys())
        time_series_flags = list(day_data.values())
        # Create a fully connected graph for simplicity
        num_sensors = len(sensor_ids)
        adjacency_matrix = np.ones((num_sensors, num_sensors)) - np.eye(num_sensors)  # Fully connected minus self-loops
        # Create the graph
        graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        # Convert to PyTorch Geometric data
        data = from_networkx(graph)
        data.x = torch.tensor(np.array(time_series_flags), dtype=torch.float)

        optimizer.zero_grad()
        out = model(data)
        loss = dtw_loss(out, time_series_flags)
        # Convert loss to a PyTorch tensor before calling backward()
        loss_tensor = torch.tensor(loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss_tensor.item()}')

# Output node embeddings
model.eval()
node_embeddings = model.fc.weight.detach().numpy()
print(node_embeddings)

# Create a graph using the learned node embeddings
G = nx.Graph()
for i in range(num_sensors):
    for j in range(i + 1, num_sensors):
        G.add_edge(i, j, weight=euclidean(node_embeddings[i], node_embeddings[j]))

# Plot the graph
pos = nx.spring_layout(G)  # Position nodes using a spring layout
nx.draw(G, pos, with_labels=True, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()