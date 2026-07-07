import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
df = pd.read_csv("Crop_recommendation.csv")

print("--- DATASET HEAD ---")
print(df.head())
print("\n--- DATASET SHAPE ---")
print(df.shape)
print("\n--- DATASET INFO ---")
print(df.info())
print("\n--- NULL VALUE CHECK ---")
print(df.isnull().sum())

# Separate Input and Output
X = df.drop("label", axis=1)
y = df["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Check Accuracy
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print("\n--- MODEL ACCURACY ---")
print(f"Accuracy: {accuracy}")

# Save Model
joblib.dump(model, "model.pkl")
print("\nModel Saved Successfully")