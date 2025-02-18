import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Dataset
df = pd.read_csv("relationship_data.csv")  # Ensure this file has Relationship_Length column

# 2. Define Features (X) and Target (y)
X = df.drop(columns=["Relationship_Length"])  # Features (everything except target)
y = df["Relationship_Length"]  # Target variable (numeric)

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Regression Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate Model
mae = mean_absolute_error(y_test, y_pred)  # Measures average prediction error
r2 = r2_score(y_test, y_pred)  # Measures how well the model explains variability

print(f"Mean Absolute Error: {mae:.2f} months")
print(f"R² Score: {r2:.2f}")  # Closer to 1 means better prediction

# DRIVING AREA
new_relationship = pd.DataFrame({
    "L_A": [10],  # How much A likes B (1-10)
    "L_B": [7],  # How much B likes A (1-10)
    "Communication": [3],  # Communication quality (1-10)
    "Trust": [3],  # Trust level (1-5)
    "Conflict": [6]  # Conflict frequency (1-10)
})

# Predict Relationship Length
predicted_length = model.predict(new_relationship)[0]
print(f"Expected Relationship Length: {predicted_length:.2f} months")