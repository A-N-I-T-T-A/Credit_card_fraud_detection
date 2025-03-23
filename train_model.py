import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("credit_card_fraud.csv")

# Features and Target
X = data.drop(columns=['Fraud_Label'])
y = data['Fraud_Label']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate Model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save Model and Scaler
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save Accuracy
with open("model_accuracy.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%")

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Model and scaler saved successfully!")
