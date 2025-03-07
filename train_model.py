import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("fishing_data.csv")

# Features & Target
X = df[['latitude', 'longitude', 'vessel_speed', 'proximity_to_protected_area']]
y = df['illegal_fishing']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open("fishing_model.pkl", "wb") as f:
    pickle.dump(model, f)