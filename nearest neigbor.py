import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your data (replace 'your_data.csv' with the actual path to your CSV file)
file_path = 'heart.csv'
df = pd.read_csv(file_path)

# Identify the target variable explicitly
target_column = 'target' # Replace 'target' with the actual name of your target variable

# X contains all columns except the target variable
X = df.iloc[:, :-1]

# y is the target variable
y = df.iloc[:,-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNeighborsClassifier (you can adjust 'n_neighbors' as needed)
knn_classifier = KNeighborsClassifier(n_neighbors=50)

# Train the classifier on the training set
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Precision
precision = precision_score(y_test, predictions, average='weighted')
print(f"Precision: {precision}")

# Recall
recall = recall_score(y_test, predictions, average='weighted')
print(f"Recall: {recall}")

# F1 Score
f1 = f1_score(y_test, predictions, average='weighted')
print(f"F1 Score: {f1}")