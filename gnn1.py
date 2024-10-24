import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
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

def dtw_distance(ts_a, ts_b):
    len_a, len_b = len(ts_a), len(ts_b)
    dtw_matrix = np.zeros((len_a + 1, len_b + 1))
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 1:] = np.inf

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = abs(ts_a[i-1] - ts_b[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                          dtw_matrix[i, j-1],    # deletion
                                          dtw_matrix[i-1, j-1])  # match
    return dtw_matrix[len_a, len_b]

def build_graph_for_day(day_data, day_labels):
    nodes = list(day_data.keys())
    # edges = []
    # edge_weights = []

    node_features = []
    node_labels = []

    for node in nodes:
        ts_a = day_data[node]
        node_features.append(ts_a)  # Time series as features
        node_labels.append(1 if day_labels[node] != "normal" else 0)  # Binary label: 1 for anomaly, 0 for normal
    
    # for i, node_a in enumerate(nodes):
    #     ts_a = day_data[node_a]
    #     node_features.append(ts_a)  # Time series as features
    #     node_labels.append(1 if day_labels[node_a] != "normal" else 0)  # Binary label: 1 for anomaly, 0 for normal
        
    #     for j, node_b in enumerate(nodes):
    #         if i < j:
    #             ts_b = day_data[node_b]
    #             distance = dtw_distance(ts_a, ts_b)
    #             edges.append((i, j))
    #             edge_weights.append(distance)
    
    # edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    edge_index = torch.tensor(
        [[i, j] for i in range(len(nodes)) for j in range(len(nodes)) if i != j], dtype=torch.long).t().contiguous()
    
    x = torch.tensor(np.array(node_features), dtype=torch.float)  # Time series data as node features
    y = torch.tensor(node_labels, dtype=torch.long)  # Anomaly labels as target
    
    # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print("Node Labels:", node_labels)
    return Data(x=x, edge_index=edge_index, y=y)

# Build Graph
day_7_graph = build_graph_for_day(data[7], labels[7])

# Split data for training
transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)
train_data, val_data, test_data = transform(day_7_graph)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Softmax for classification

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    # def decode(self, z, pos_edge_index, neg_edge_index):
    #     edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    #     return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

model = GNN(in_channels=day_7_graph.num_node_features, out_channels=2)

# # Prepare data for training
# data = train_test_split_edges(day_7_graph)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()

    z = model(train_data.x, train_data.edge_index)  # Forward pass

    # Check the output shape and print it
    print(f"Output shape: {z.shape}")

    # Print the labels to check if they're None (This was an error I was having)
    # if train_data.y is None:
    #     print("Warning: train_data.y is None")
    # else:
    #     print(f"Labels shape: {train_data.y.shape}")
    #     print(f"Labels: {train_data.y}")
    
    loss = criterion(z, train_data.y)  # Cross-entropy loss for classification

    # z = model.encode(train_data.x, train_data.edge_index)
    
    # pos_pred = model.decode(z, train_data.pos_edge_index, train_data.neg_edge_index)
    # neg_pred = model.decode(z, train_data.neg_edge_index, train_data.neg_edge_index)
    
    # pos_loss = -torch.log(pos_pred + 1e-15).mean()
    # neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
    # loss = pos_loss + neg_loss
    
    loss.backward()
    optimizer.step()
    return loss

for epoch in range(1, 101):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    z = model(test_data.x, test_data.edge_index)
    pred = z.argmax(dim=1)  # Get predicted labels
    accuracy = (pred == test_data.y).sum().item() / len(test_data.y)
    print(f'Accuracy: {accuracy}')
    # z = model.encode(test_data.x, test_data.edge_index)
    # pos_pred = model.decode(z, test_data.pos_edge_index, test_data.neg_edge_index)
    # neg_pred = model.decode(z, test_data.neg_edge_index, test_data.neg_edge_index)