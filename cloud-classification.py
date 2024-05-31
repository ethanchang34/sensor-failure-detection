from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def classify_anomaly_type(normal_data, anomaly_data, anomaly_types):
  """
  Classifies the type of anomaly in sensor data using supervised learning.

  Args:
      normal_data: A numpy array of sensor data points representing normal behavior.
      anomaly_data: A numpy array of sensor data points marked as anomalous.
      anomaly_types: A list of strings representing different types of anomalies 
                     (e.g., ["Spike", "Dip", "Drift"]).

  Returns:
      A trained RandomForestClassifier model.
  """

  # Feature engineering (optional)
  # You might want to extract additional features from the sensor data 
  # that could be helpful for classification (e.g., rolling averages, standard deviation)

  # Combine data and labels
  X = np.concatenate((normal_data, anomaly_data))
  y = np.concatenate(([0] * len(normal_data), [1] * len(anomaly_data)))
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train a RandomForestClassifier model (replace with other classifiers if needed)
  model = RandomForestClassifier(n_estimators=100, random_state=0)
  model.fit(X_train, y_train)

  # Print classification report (optional)
  from sklearn.metrics import classification_report
  print(classification_report(y_test, model.predict(X_test)))

  return model

# Example usage
# Assuming you have normal sensor data in 'normal_data' and anomalous data in 'anomaly_data'
# You also have a list of possible anomaly types in 'anomaly_types'

model = classify_anomaly_type(normal_data, anomaly_data, anomaly_types)

# Use the trained model to predict the type of anomaly for new data points
new_data = ...  # Your new sensor data point
predicted_type = model.predict([new_data])[0]
print("Predicted anomaly type:", anomaly_types[predicted_type])