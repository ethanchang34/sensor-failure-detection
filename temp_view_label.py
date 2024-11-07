import pickle

with open('Flagged Data/labels.pkl', 'rb') as file:
    labels = pickle.load(file)

def count(dict):
    count = 0
    for sensor_id, label in dict.items():
        if label == 'normal':
            count += 1
    return count

print(count(labels[15]))
print(count(labels[23]))
print(count(labels[7]))
print(count(labels[25]))
print(count(labels[18]))
20
22
19
21
20