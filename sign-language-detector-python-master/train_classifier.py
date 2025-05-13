import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('data.pickle', 'rb'))

# Ensure uniform shape
data = data_dict['data']
labels = data_dict['labels']

# Check the shape of each element
lengths = [len(d) if hasattr(d, '__len__') else 0 for d in data]
expected_length = max(set(lengths), key=lengths.count)  # most common length

# Filter out inconsistent samples
filtered_data = []
filtered_labels = []

for d, l in zip(data, labels):
    if hasattr(d, '__len__') and len(d) == expected_length:
        filtered_data.append(d)
        filtered_labels.append(l)

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
