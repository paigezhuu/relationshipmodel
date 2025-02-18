from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

# Load dataset
df = pd.read_csv("relationship_data.csv")

# Define Features (X) and Target (y)
X = df.drop(columns=["Relationship_Length"])  # Features (everything except target)
y = df["Relationship_Length"]  # Target variable (numeric)

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Regression Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_data', methods=['POST'])
def submit_data():
    # Get data from form
    L_A = int(request.form['L_A'])
    L_B = int(request.form['L_B'])
    Communication = int(request.form['Communication'])
    Trust = int(request.form['Trust'])
    Conflict = int(request.form['Conflict'])
    
    # Add new data to CSV
    new_data = pd.DataFrame({
        'L_A': [L_A], 
        'L_B': [L_B],
        'Communication': [Communication],
        'Trust': [Trust],
        'Conflict': [Conflict]
    })
    new_data.to_csv("relationship_data.csv", mode='a', header=False, index=False)
    
    # Update model
    df = pd.read_csv("relationship_data.csv")
    X = df.drop(columns=["Relationship_Length"])
    y = df["Relationship_Length"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediction
    new_relationship = pd.DataFrame({
        "L_A": [L_A],
        "L_B": [L_B],
        "Communication": [Communication],
        "Trust": [Trust],
        "Conflict": [Conflict]
    })
    predicted_length = model.predict(new_relationship)[0]
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return jsonify({
        'predicted_length': predicted_length,
        'mae': mae,
        'r2': r2
    })

if __name__ == "__main__":
    app.run(debug=True)
