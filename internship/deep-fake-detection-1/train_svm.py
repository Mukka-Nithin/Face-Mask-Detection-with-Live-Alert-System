import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Load embeddings
data = np.load('embeddings/embeddings.npz')
X, y = data['X'], data['y']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy with RandomForest: {acc:.2f}")

# Save model
joblib.dump(rf_model, 'svm_model.pkl')
