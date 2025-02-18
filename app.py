from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

# Load Dataset
df = pd.read_csv("relationship_data.csv")

# Define Features (X) and Target (y)
X = df.drop(columns=["Relationship_Length"])  # Features (everything except target)
y = df["Relationship_Length"]  # Target variable (numeric)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input_data', methods=['POST'])
def input_data():
    # Get data from form
    L_A = int(request.form['L_A'])
    L_B = int(request.form['L_B'])
    Communication = int(request.form['Communication'])
    Trust = int(request.form['Trust'])
    Conflict = int(request.form['Conflict'])
    Relationship_Length = int(request.form['Relationship_Length'])

    # Add new data to CSV
    new_data = pd.DataFrame({
        'L_A': [L_A], 
        'L_B': [L_B],
        'Communication': [Communication],
        'Trust': [Trust],
        'Conflict': [Conflict],
        'Relationship_Length': [Relationship_Length]
    })
    new_data.to_csv("relationship_data.csv", mode='a', header=False, index=False)

    # Update model
    df = pd.read_csv("relationship_data.csv")
    X = df.drop(columns=["Relationship_Length"])
    y = df["Relationship_Length"]
    model.fit(X, y)

    return jsonify({"message": "Data added successfully! The model has been updated."})

@app.route('/predict', methods=['POST'])
def predict():
    # Get data for prediction
    L_A = int(request.form['L_A'])
    L_B = int(request.form['L_B'])
    Communication = int(request.form['Communication'])
    Trust = int(request.form['Trust'])
    Conflict = int(request.form['Conflict'])

    # Prepare the input data for prediction
    new_relationship = pd.DataFrame({
        "L_A": [L_A],
        "L_B": [L_B],
        "Communication": [Communication],
        "Trust": [Trust],
        "Conflict": [Conflict]
    })

    # Make Prediction
    predicted_length = model.predict(new_relationship)[0]
    
    return jsonify({
        'predicted_length': predicted_length
    })

if __name__ == "__main__":
    app.run(debug=True)
