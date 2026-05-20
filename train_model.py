import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset/creditcard_reduced.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model/fraud_model.pkl", "wb"))

# Feature importance
importance = model.feature_importances_

features = X.columns

# Create dataframe
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(feature_importance.head(10))

# Plot graph
plt.figure(figsize=(10,6))
plt.barh(feature_importance["Feature"][:10],
         feature_importance["Importance"][:10])

plt.title("Top Features for Fraud Detection")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()
