import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from sklearn.model_selection import train_test_split
import pickle


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# print(f"Using device: {device}")

with open('Flagged Data/flagged_data.pkl', 'rb') as file:
    flagged_data = pickle.load(file)

with open('Flagged Data/speed_dict_with_anomaly.pkl', 'rb') as file:
    data = pickle.load(file)

with open('Flagged Data/clustered_sensors.pkl', 'rb') as file:
    clustered_sensors = pickle.load(file)

with open('Flagged Data/labels.pkl', 'rb') as file:
    labels = pickle.load(file)

print(clustered_sensors)  # Dictionary of sensor IDs and cluster labels

# Initialize dictionaries to store tensors and mappings
tensor_data = {}
sensor_id_mapping = {}
# Convert each day's sensor data to tensors and retain mappings
for day, sensors in data.items():
    sensor_ids = list(sensors.keys())
    sensor_id_mapping[day] = sensor_ids  # Retain sensor ID mapping for each day
    sensor_values = [sensors[sensor_id] for sensor_id in sensor_ids]
    tensor_data[day] = torch.tensor(sensor_values, dtype=torch.float)

# Initialize dictionaries to store tensors and mappings
bin_labels = {}
label_mapping = {}
# Convert labeling to 0 for normal and 1 for anomaly
for day, sensors in labels.items():
    sensor_ids = list(sensors.keys())
    label_mapping[day] = sensor_ids
    day_labels = [0 if sensors[sensor_id] == "normal" else 1 for sensor_id in sensor_ids]
    bin_labels[day] = torch.tensor(day_labels, dtype=torch.float)

# Ensure that the sensor_id_mappign and label_mapping are the same, meaning the x aligns with the correct y
# print(sensor_id_mapping[7])
# print(label_mapping[7])


# def build_graph_for_day(day_data, day_labels):
#     nodes = list(day_data.keys())
#     # edges = []
#     # edge_weights = []

#     node_features = []
#     node_labels = []

#     for node in nodes:
#         ts_a = day_data[node]
#         node_features.append(ts_a)  # Time series as features
#         node_labels.append(1 if day_labels[node] != "normal" else 0)  # Binary label: 1 for anomaly, 0 for normal
    
#     # for i, node_a in enumerate(nodes):
#     #     ts_a = day_data[node_a]
#     #     node_features.append(ts_a)  # Time series as features
#     #     node_labels.append(1 if day_labels[node_a] != "normal" else 0)  # Binary label: 1 for anomaly, 0 for normal
        
#     #     for j, node_b in enumerate(nodes):
#     #         if i < j:
#     #             ts_b = day_data[node_b]
#     #             distance = dtw_distance(ts_a, ts_b)
#     #             edges.append((i, j))
#     #             edge_weights.append(distance)
    
#     # edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     # edge_attr = torch.tensor(edge_weights, dtype=torch.float)

#     edge_index = torch.tensor(
#         [[i, j] for i in range(len(nodes)) for j in range(len(nodes)) if i != j], dtype=torch.long).t().contiguous()
    
#     x = torch.tensor(np.array(node_features), dtype=torch.float)  # Time series data as node features
#     y = torch.tensor(node_labels, dtype=torch.long)  # Anomaly labels as target
    
#     # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
#     print("Node Labels:", node_labels)
#     return Data(x=x, edge_index=edge_index, y=y)

# # Build Graph
# day_7_graph = build_graph_for_day(data[7], labels[7])

# Prepare the graph structure once (only nodes and edges)
num_nodes = len(data[7])  # Replace with the actual number of nodes in your graph
edge_index = torch.tensor(
    [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
    dtype=torch.long
).t().contiguous()
graph_structure = Data(edge_index=edge_index)  # Use only fixed edge_index

# Split the keys (days) into train, val, test splits
day_keys = list(data.keys())
train_days, test_days = train_test_split(day_keys, test_size=0.2, random_state=42)
train_days, val_days = train_test_split(train_days, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# day = 7
# print(f"Tensor for Day {day}:", tensor_data[day])  # Tensor with sensor data
# print(f"Tensor shape: {tensor_data[day].shape}")
# print(f"Sensor IDs for Day {day}:", sensor_id_mapping[day])  # Original sensor IDs
# print(tensor_data[7])
# print(bin_labels[7])

# Create subsets based on day splits
train_data = {day: (tensor_data[day], bin_labels[day]) for day in train_days}
val_data = {val_day: (tensor_data[val_day], bin_labels[val_day]) for val_day in val_days}
test_data = {test_day: (tensor_data[test_day], bin_labels[test_day]) for test_day in test_days}
print(f"Train days: {len(train_data)} | Validation days: {len(val_data)} | Test days: {len(test_data)}")
print("Training data days: ", train_data.keys())
print("Validation data days: ", val_data.keys())
print("Test data days: ", test_data.keys())

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Softmax for classification

    # def encode(self, x, edge_index):
    #     x = F.relu(self.conv1(x, edge_index))
    #     return self.conv2(x, edge_index)

    # def decode(self, z, pos_edge_index, neg_edge_index):
    #     edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    #     return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

sample_day = data[7]
first_node = next(iter(sample_day.values()))  # Get the time series of the first node
num_node_features = len(first_node)  # Number of features for each nod
# model = GNN(in_channels=day_7_graph.num_node_features, out_channels=2)
model = GNN(in_channels=num_node_features, out_channels=2)
# # Get the number of features from a sample graph
# sample_day = next(iter(train_data))  # Get an arbitrary key from train_data
# num_node_features = train_data[sample_day]['x'].shape[1]  # Assuming 'x' holds node features

# # Initialize model with the correct number of input channels
# model = GNN(in_channels=num_node_features, out_channels=2)

# # Prepare data for training
# data = train_test_split_edges(day_7_graph)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# def train():
#     model.train()
#     optimizer.zero_grad()

#     z = model(train_data.x, train_data.edge_index)  # Forward pass

#     # Check the output shape and print it
#     print(f"Output shape: {z.shape}")

#     # Print the labels to check if they're None (This was an error I was having)
#     # if train_data.y is None:
#     #     print("Warning: train_data.y is None")
#     # else:
#     #     print(f"Labels shape: {train_data.y.shape}")
#     #     print(f"Labels: {train_data.y}")
    
#     loss = criterion(z, train_data.y)  # Cross-entropy loss for classification

#     # z = model.encode(train_data.x, train_data.edge_index)
    
#     # pos_pred = model.decode(z, train_data.pos_edge_index, train_data.neg_edge_index)
#     # neg_pred = model.decode(z, train_data.neg_edge_index, train_data.neg_edge_index)
    
#     # pos_loss = -torch.log(pos_pred + 1e-15).mean()
#     # neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
#     # loss = pos_loss + neg_loss
    
#     loss.backward()
#     optimizer.step()
#     return loss

# Training function
def train_one_day(day_features, day_label):
    model.train()
    optimizer.zero_grad()

    graph_structure.x = day_features  # Update node features for the day's data
    z = model(graph_structure.x, graph_structure.edge_index)  # Forward pass

    loss = criterion(z, torch.tensor([day_label]))
    loss.backward()
    optimizer.step()

    return loss.item()

# for epoch in range(1, 101):
#     loss = train()
#     # print(f'Epoch {epoch}, Loss: {loss.item()}')
#     print(f'Epoch {epoch}, Loss: {loss}')

# Training over multiple days
for epoch in range(1,101):
    for day, (features, bin_labels) in train_data.items():
        loss = train_one_day(features, bin_labels)
        print(f'Epoch {epoch}, Day {day}, Loss: {loss}')

# model.eval()

# Validation
def evaluate(data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)  # Forward pass
        pred = z.argmax(dim=1)  # Get predicted labels
        accuracy = (pred == data.y).sum().item() / len(data.y)
    return accuracy

# Testing function
def test_one_day(day_features, day_label):
    model.eval()
    with torch.no_grad():
        graph_structure.x = day_features  # Update with test day features
        z = model(graph_structure.x, graph_structure.edge_index)
        pred = z.argmax(dim=1).item()  # Predicted label for the day

        return pred == day_label  # True if prediction matches label

# Testing over test days
accuracy = sum(test_one_day(features, label) for features, label in test_data.values()) / len(test_data)
print(f'Overall Test Accuracy: {accuracy}')

# # Calculate validation and test accuracy
# val_accuracy = evaluate(val_data)
# test_accuracy = evaluate(test_data)
# print(f'Validation Accuracy: {val_accuracy}')
# print(f'Test Accuracy: {test_accuracy}')

# with torch.no_grad():
#     z = model(test_data.x, test_data.edge_index)
#     pred = z.argmax(dim=1)  # Get predicted labels
#     accuracy = (pred == test_data.y).sum().item() / len(test_data.y)
#     print(f'Accuracy: {accuracy}')
#     # z = model.encode(test_data.x, test_data.edge_index)
#     # pos_pred = model.decode(z, test_data.pos_edge_index, test_data.neg_edge_index)
#     # neg_pred = model.decode(z, test_data.neg_edge_index, test_data.neg_edge_index)