import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_scatter
from sklearn.model_selection import train_test_split
import pickle
from fastdtw import fastdtw


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# print(f"Using device: {device}")

with open('Flagged Data/speed_dict.pkl', 'rb') as file:
    raw = pickle.load(file)

with open('Flagged Data/flagged_with_anomaly.pkl', 'rb') as file:
    data = pickle.load(file)

# with open('Flagged Data/speed_dict_with_anomaly.pkl', 'rb') as file:
    # data = pickle.load(file)

with open('Flagged Data/clustered_sensors.pkl', 'rb') as file:
    clustered_sensors = pickle.load(file)

with open('Flagged Data/labels.pkl', 'rb') as file:
    labels = pickle.load(file)

clusters = {}
for sensor_id, cluster_label in clustered_sensors.items():
    if cluster_label not in clusters:
        clusters[cluster_label] = []
    clusters[cluster_label].append(sensor_id)
print(clusters)

# Initialize dictionaries to store tensors and mappings
tensor_data = {}
sensor_id_mapping = {}
# Convert each day's sensor data to tensors and retain mappings
for day, sensors in data.items():
    sensor_ids = list(sensors.keys())
    sensor_id_mapping[day] = sensor_ids  # Retain sensor ID mapping for each day
    sensor_values = [[sensors[sensor_id]] for sensor_id in sensor_ids]
    tensor_data[day] = torch.tensor(np.array(sensor_values), dtype=torch.float)

raw_mapping = list(raw[next(iter(raw))].keys())
# Flatten the data for DTW corr matrix
flattened_data = {sensor_id: [] for sensor_id in sensor_ids}
# tensor_raw = {}
raw_id_mapping = {}
for day, sensors in raw.items():
    sensor_ids = list(sensors.keys())
    raw_id_mapping[day] = sensor_ids
    for sensor_id, sensor_values in sensors.items():
        flattened_data[sensor_id].extend(sensor_values)
    # sensor_ids = [raw_mapping.index(sensor_id) for sensor_id in sensor_ids]
    # sensor_values = [[sensors[sensor_id]] for sensor_id in sensor_ids]
    # tensor_raw[day] = torch.tensor(np.array(sensor_values), dtype=torch.float)
flattened_data = np.array([flattened_data[sensor_id] for sensor_id in sensor_ids])
    

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
# print(raw_id_mapping[7])

# day = 7
# print(f"Tensor for Day {day}:", tensor_data[day])  # Tensor with sensor data
# print(f"Tensor Data shape: {tensor_data[day].shape}")
# print(f"Tensor Labels shape: {bin_labels[day].shape}")
# print(tensor_data[7])
# print(bin_labels[7])

# Prepare the graph structure once (only nodes and edges)
# num_nodes = len(data[7])  # Replace with the actual number of nodes in your graph
# edge_index = torch.tensor(
#     [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
#     dtype=torch.long
# ).t().contiguous()
# graph_structure = Data(edge_index=edge_index)  # Use only fixed edge_index

# Split the keys (days) into train, val, test splits
day_keys = list(data.keys())
train_days, test_days = train_test_split(day_keys, test_size=0.2, random_state=42)
train_days, val_days = train_test_split(train_days, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
print("Training data days: ", train_days)
print("Validation data days: ", val_days)
print("Test data days: ", test_days)

# Create subsets based on day splits
# train_data = {day: (tensor_data[day], bin_labels[day]) for day in train_days}
# val_data = {val_day: (tensor_data[val_day], bin_labels[val_day]) for val_day in val_days}
# test_data = {test_day: (tensor_data[test_day], bin_labels[test_day]) for test_day in test_days}
# print(f"Train days: {len(train_data)} | Validation days: {len(val_data)} | Test days: {len(test_data)}")
# print("Training data days: ", train_data.keys())
# print("Validation data days: ", val_data.keys())
# print("Test data days: ", test_data.keys())

class CustomMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomMessagePassing, self).__init__(aggr='max')  # 'max' aggregation for message computation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # x: Node feature matrix
        # edge_index: Graph connectivity in COO format with shape [2, num_edges]
        # edge_weight: Edge weights (DTW values) with shape [num_edges]
        x = self.lin(x)  # Linear transformation on the node features
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)  # Message passing
        return out

    def message(self, x_j, edge_weight):
        # x_j: Neighbor node features
        # edge_weight: Edge weights (DTW values between self and neighbor nodes)
        # Compute e * V
        return edge_weight.view(-1, 1) * x_j

    def aggregate(self, inputs, index):
        # Custom aggregation: Max aggregation over neighbors
        return torch_scatter.scatter_max(inputs, index, dim=0)[0]

    def update(self, aggr_out, x):
        # Custom update function to apply formula min(self, self - max(e*V))
        return torch.min(x, x - aggr_out)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        # self.conv1 = GCNConv(in_channels, 16)
        # self.conv2 = GCNConv(16, out_channels)
        # self.conv1 = GCNConv(in_channels, out_channels)
        self.conv1 = CustomMessagePassing(in_channels, 16)
        self.conv2 = CustomMessagePassing(16, out_channels)
        # self.dropout = torch.nn.Dropout(p=0.3)

    # def forward(self, x, edge_index):
    #     x = F.relu(self.conv1(x, edge_index))
    #     # x = self.dropout(x)  # Add dropout for regularization
    #     # x = self.conv2(x, edge_index)
    #     return F.log_softmax(x, dim=1)  # Softmax for classification

    def forward(self, x, edge_index, edge_weight):
        # Forward pass with custom message passing layer
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)  # Softmax for classification

    # def encode(self, x, edge_index):
    #     x = F.relu(self.conv1(x, edge_index))
    #     return self.conv2(x, edge_index)

    # def decode(self, z, pos_edge_index, neg_edge_index):
    #     edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    #     return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

def compute_anomaly_metrics(preds, labels):
    true_positives = ((preds == 1) & (labels == 1)).sum().item()
    false_positives = ((preds == 1) & (labels == 0)).sum().item()
    false_negatives = ((preds == 0) & (labels == 1)).sum().item()
    true_negatives = ((preds == 0) & (labels == 0)).sum().item()

    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1_score
'''
# sample_day = data[7]
# first_node = next(iter(sample_day.values()))  # Get the time series of the first node
# num_node_features = len(first_node)  # Number of features for each node
# Initialize model with the correct number of input channels
# model = GNN(in_channels=num_node_features, out_channels=2)
model = GNN(in_channels=1, out_channels=2)

# Count normal and anomaly labels in the training data to determine class weights
normal_count = sum((labels == 0).sum().item() for _, labels in train_data.values())
anomaly_count = sum((labels == 1).sum().item() for _, labels in train_data.values())

# Calculate the anomaly ratio to determine weights
normal_weight = len([label for day in train_data.values() for label in day[1] if label == 0])
anomaly_weight = len([label for day in train_data.values() for label in day[1] if label == 1])
total_weight = normal_weight + anomaly_weight

# Inverse weights to prioritize anomalies (higher weight)
weights = torch.tensor([1.0, total_weight / anomaly_weight])
criterion = torch.nn.CrossEntropyLoss(weight=weights)

# Optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.CrossEntropyLoss()
'''
def threshold_predictions(logits, threshold=0.5):
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)
    preds = torch.zeros_like(logits[:, 0], dtype=torch.long)
    preds[probs[:, 1] > threshold] = 1  # Predict anomaly if probability > threshold
    return preds

# Training function
def train_one_day(day_features, day_labels, edge_weights):
    model.train()
    optimizer.zero_grad()

    graph_structure.x = day_features  # Update node features for the day's data
    z = model(graph_structure.x, graph_structure.edge_index, edge_weights)  # Forward pass
    # z = torch.sigmoid(model(graph_structure.x, graph_structure.edge_index))

    day_labels = day_labels.long()
    loss = criterion(z, day_labels)
    loss.backward()
    optimizer.step()

    # Calculate predictions and anomaly metrics
    preds = z.argmax(dim=1)
    # preds = threshold_predictions(z)
    accuracy, precision, recall, f1_score = compute_anomaly_metrics(preds, day_labels)

    # print(f"Loss: {loss.item():.4f} | Acc: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f}")

    return loss.item()
'''
# Training over multiple days
for epoch in range(1,101):
    for day, (features, day_bin_labels) in train_data.items():
        loss = train_one_day(features, day_bin_labels)
        # print(f'Epoch {epoch}, Day {day}, Loss: {loss}')
'''
# Testing function
def test_one_day(day_features, day_labels, edge_weights):
    model.eval()
    with torch.no_grad():
        graph_structure.x = day_features  # Update with test day features
        z = model(graph_structure.x, graph_structure.edge_index, edge_weights)

        # Convert day_labels to a tensor if it's not already
        if not isinstance(day_labels, torch.Tensor):
            day_labels = torch.tensor(day_labels, dtype=torch.long)  # Ensure day_labels is a tensor

        preds = z.argmax(dim=1)  # Predicted labels for the day
        # preds = threshold_predictions(z)
        correct = preds.eq(day_labels).sum().item()  # Count correct predictions
        accuracy = correct / len(day_labels)  # Accuracy for this day

        # Calculate anomaly-focused metrics
        accuracy, precision, recall, f1_score = compute_anomaly_metrics(preds, day_labels)
        print(f"Test - Acc: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f}")
        print(f"Acc: {accuracy*len(day_labels):.4f}")

        return accuracy
'''
# Testing over test days
accuracy = sum(test_one_day(features, label) for features, label in test_data.values()) / len(test_data)
print(f'Overall Test Accuracy: {accuracy}')
'''
for cluster_label, sensor_ids in clusters.items():
    print("Cluster: ", cluster_label)
    if not sensor_ids:
        continue
    
    
    day_data_for_cluster = {
        day: torch.stack([tensor_data[day][sensor_id_mapping[day].index(sensor_id)] for sensor_id in sensor_ids]) for day, day_data in tensor_data.items()
    }
    bin_labels_for_cluster = {
        day: torch.stack([bin_labels[day][sensor_id_mapping[day].index(sensor_id)] for sensor_id in sensor_ids]) for day, day_labels in bin_labels.items()
    }

    num_cluster_nodes = len(sensor_ids)

    dtw_corrs = np.zeros([num_cluster_nodes, num_cluster_nodes])
    for i in range(num_cluster_nodes):
        for j in range(i+1, num_cluster_nodes):
            if i == j:
                dtw_corrs[i, j] = 0
            else:
                distance, _ = fastdtw(flattened_data[i], flattened_data[j])
                dtw_corrs[i, j] = distance
                dtw_corrs[j, i] = distance
    dtw_corrs = torch.tensor(dtw_corrs, dtype=torch.float)
    max_value = torch.max(dtw_corrs).item()
    dtw_corrs_norm = dtw_corrs / max_value
    # print(dtw_corrs)
    # print(dtw_corrs_norm)

    edge_weights = torch.tensor(
    [dtw_corrs_norm[i, j] for i in range(num_cluster_nodes) for j in range(num_cluster_nodes) if i != j],
    dtype=torch.float
    )

    edge_index = torch.tensor(
        [[i, j] for i in range(num_cluster_nodes) for j in range(num_cluster_nodes) if i != j],
        dtype=torch.long
    ).t().contiguous()
    graph_structure = Data(edge_index=edge_index)  # Use only the cluster-specific edge_index

    cluster_train_data = {day: (day_data_for_cluster[day], bin_labels_for_cluster[day]) for day in train_days}
    cluster_val_data = {val_day: (day_data_for_cluster[val_day], bin_labels_for_cluster[val_day]) for val_day in val_days}
    cluster_test_data = {test_day: (day_data_for_cluster[test_day], bin_labels_for_cluster[test_day]) for test_day in test_days}

    model = GNN(in_channels=1, out_channels=2)

    # Count normal and anomaly labels in the training data to determine class weights
    normal_count = sum((labels == 0).sum().item() for _, labels in cluster_train_data.values())
    anomaly_count = sum((labels == 1).sum().item() for _, labels in cluster_train_data.values())

    total_count = normal_count + anomaly_count
    normal_weight = total_count / normal_count
    anomaly_weight = total_count / anomaly_count
    weights = torch.tensor([normal_weight, anomaly_weight], dtype=torch.float32)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights[1])

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training over multiple days
    for epoch in range(1,100):
        for day, (features, day_bin_labels) in cluster_train_data.items():
            loss = train_one_day(features, day_bin_labels, edge_weights)
            # print(f'Epoch {epoch}, Day {day}, Loss: {loss}')
            
        # Update the graph structure with the day's features
        graph_structure.x = features

    # Testing over test days
    accuracy = sum(test_one_day(features, label, edge_weights) for features, label in cluster_test_data.values()) / len(cluster_test_data)
    print(f'Overall Test Accuracy: {accuracy}')