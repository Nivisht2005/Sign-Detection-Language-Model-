import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Ensure all sequences in data have the same length
consistent_length = len(data_dict['data'][0])
data = [np.array(item) for item in data_dict['data'] if len(item) == consistent_length]
labels = [label for idx, label in enumerate(data_dict['labels']) if len(data_dict['data'][idx]) == consistent_length]

# Convert data and labels to NumPy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'model.p'")
