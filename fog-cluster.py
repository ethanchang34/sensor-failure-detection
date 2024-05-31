import numpy as np
import pandas as pd
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import Data
df = pd.read_csv("DelDOT Data/19912_NB.csv")
print(df.head()) # grabs first n rows (default n=5)

def read_h5_file_to_numpy(file_path, dataset_name):
    """
    Reads an H5 file and stores the specified dataset as a 2D numpy array.

    Parameters:
        file_path (str): The file path to the .h5 file.
        dataset_name (str): The name of the dataset within the .h5 file to read.

    Returns:
        np.ndarray: A 2D numpy array containing the data from the specified dataset.
    """
    # Open the H5 file
    with h5py.File(file_path, 'r') as file:
        # Make sure the dataset exists in the file
        if dataset_name in file:
            data = file[dataset_name][()]
        else:
            raise ValueError(f"Dataset {dataset_name} not found in file.")
        
        # Check if the dataset is 2D
        if data.ndim != 2:
            raise ValueError("Dataset is not 2D.")
        
        return data

# Example usage
file_path = 'your_file.h5'
dataset_name = 'your_dataset_name'
data_matrix = read_h5_file_to_numpy(file_path, dataset_name)
print(data_matrix)

'''
def batch_clustering_anomaly(data, n_clusters):
  """
  Performs batch clustering and anomaly detection on traffic data.

  My understanding: Partitions data into clusters based on similar average traffic speeds. It'll either
  identify sensors that lack a good fit to any group or identify if a sensor starts to stray from its group.
  Not sure either is the perfect answer we are looking for

  Questions: How to determine the number of clusters (something I must define)

  Args:
      data: A numpy array of shape (n_samples, n_sensors) where each row represents
            average speed data for n_minutes from n_sensors.
      n_clusters: The desired number of clusters.

  Returns:
      clusters: A list of lists, where each inner list contains indices of sensors 
                belonging to a particular cluster.
      anomaly_sensor: The index of the sensor identified as an anomaly.
      silhouette_scores: A list of silhouette scores for each cluster.
  """

  # Preprocessing
  # Standardize or normalize the data here

  # Batch clustering
  kmeans = KMeans(n_clusters=n_clusters, random_state=0)
  kmeans.fit(data)
  clusters = [[] for _ in range(n_clusters)]
  for i, label in enumerate(kmeans.labels_):
    clusters[label].append(i)

  # Anomaly detection using silhouette score
  # The silhouette score is a metric used to evaluate the quality of clustering in machine learning.
  # It considers two aspects of clustering:
  #   Cohesion: How similar data points are within a cluster (tightly packed).
  #   Separation: How different data points are between clusters (well-separated).
  #   Silhouette score = (b - a) / max(a, b)    [Ranges from -1 to 1] where (a) is own cluster and (b) is other cluster
  silhouette_scores = []
  anomaly_sensor = None
  anomaly_score = 0
  for cluster_idx, cluster in enumerate(clusters):
    if not cluster:
      continue  # Skip empty clusters
    silhouette_score_cluster = silhouette_score(data[cluster, :], kmeans.labels_[cluster])
    silhouette_scores.append(silhouette_score_cluster)
    # Identify sensor with lowest silhouette score (might be an outlier)
    if silhouette_score_cluster < anomaly_score:
      anomaly_score = silhouette_score_cluster
      anomaly_sensor = cluster[0]  # Assuming single anomaly per iteration

  return clusters, anomaly_sensor, silhouette_scores

# Example usage
# Assuming you have your traffic data in a numpy array called 'traffic_data'
n_sensors = traffic_data.shape[1]  # Number of sensors
n_clusters = 3  # Desired number of clusters (adjust as needed)

clusters, anomaly_sensor, silhouette_scores = batch_clustering_anomaly(traffic_data, n_clusters)

# Print results
print("Clusters:", clusters)
print("Anomaly sensor:", anomaly_sensor if anomaly_sensor is not None else "No anomaly detected")
print("Silhouette scores:", silhouette_scores)

'''