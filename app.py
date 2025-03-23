from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load model accuracy
with open("model_accuracy.txt", "r") as f:
    accuracy = f.read().strip().split(": ")[1]

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = np.array([
            float(request.form['Transaction_Amount']),
            float(request.form['Transaction_Time']),
            float(request.form['Previous_Transactions']),
            float(request.form['Location_Risk'])
        ]).reshape(1, -1)  # Convert to NumPy array
        
        # Preprocess input
        features_scaled = scaler.transform(features)  # Scale input
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        
        return render_template('index.html', prediction_text=f'Transaction is {result}', accuracy=accuracy)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)