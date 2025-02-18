import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 500 rows of synthetic data
num_samples = 500
data = {
    "L_A": np.random.randint(1, 11, num_samples),
    "L_B": np.random.randint(1, 11, num_samples),
    "Communication": np.random.randint(1, 11, num_samples),
    "Trust": np.random.randint(1, 6, num_samples),
    "Conflict": np.random.randint(1, 11, num_samples)
}

# Relationship length formula (simulated)
# Higher L_A, L_B, Communication, Trust = longer relationship
# Higher Conflict = shorter relationship
#data["Relationship_Length"] = ((
#    2 * (data["L_A"] + data["L_B"]) +
#    1 * data["Communication"] +
#    3 * data["Trust"] -
#    2 * data["Conflict"] +
#    np.random.normal(0, 5, num_samples) // 15)  # Adds some randomness
#)

data["Relationship_Length"] = np.random.randint(1,60,num_samples)

# Ensure all relationship lengths are at least 1 month
data["Relationship_Length"] = np.maximum(1, data["Relationship_Length"]).astype(int)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("relationship_data.csv", index=False)

print("Dataset generated and saved as 'relationship_data.csv'.")