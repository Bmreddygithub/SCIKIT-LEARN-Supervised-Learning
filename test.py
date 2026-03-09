"""
Simple Tests for Heart Disease Prediction System
Run with: python tests.py
"""

import numpy as np
from data_loader import load_heart_disease_data
from sklearn.ensemble import RandomForestClassifier

# ── Load data and train model once for all tests ──────────────────────────────
print("Setting up: loading data and training model...")
X_train, X_test, y_train, y_test, feature_names, scaler = load_heart_disease_data()
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
print("Setup complete.\n")
print("=" * 50)

# ── TEST 1: Predict a known disease-positive patient from the test set ─────────
print("\nTEST 1: Known true +ve patient prediction")
print("-" * 50)

# Pick the first test patient who actually has disease (y_test == 1)
disease_indices = np.where(y_test == 1)[0]
idx = disease_indices[0]
patient = X_test[idx].reshape(1, -1)  # already scaled

prediction = model.predict(patient)[0]
confidence = model.predict_proba(patient)[0]
actual = y_test[idx]

print(f"Input:      Test patient #{idx} (actual label: DISEASE)")
print(f"Prediction: {'DISEASE DETECTED' if prediction == 1 else 'NO DISEASE'}")
print(f"Confidence: No Disease={confidence[0]*100:.1f}%  |  Disease={confidence[1]*100:.1f}%")
print(f"Result:     {'PASS' if prediction == actual else 'FAIL - model missed a disease case'}")

# ── TEST 2: Predict a known healthy patient from the test set ─────────────────
print("\nTEST 2: Known healthy patient prediction")
print("-" * 50)

# Pick the first test patient who is actually healthy (y_test == 0)
healthy_indices = np.where(y_test == 0)[0]
idx2 = healthy_indices[0]
patient2 = X_test[idx2].reshape(1, -1)  # already scaled

prediction = model.predict(patient2)[0]
confidence = model.predict_proba(patient2)[0]
actual = y_test[idx2]

print(f"Input:      Test patient #{idx2} (actual label: NO DISEASE)")
print(f"Prediction: {'DISEASE DETECTED' if prediction == 1 else 'NO DISEASE'}")
print(f"Confidence: No Disease={confidence[0]*100:.1f}%  |  Disease={confidence[1]*100:.1f}%")
print(f"Result:     {'PASS' if prediction == actual else 'FAIL - model gave false alarm'}")

# ── TEST 3: Overall accuracy on the test set ──────────────────────────────────
print("\nTEST 3: Overall accuracy on held-out test set")
print("-" * 50)

y_pred = model.predict(X_test)
correct = int((y_pred == y_test).sum())
total = len(y_test)
accuracy = correct / total * 100

print(f"Input:    {total} unseen patient records from the test set")
print(f"Correct:  {correct} out of {total}")
print(f"Accuracy: {accuracy:.1f}%")
print(f"Result:   {'PASS' if accuracy >= 75 else 'FAIL - accuracy below 75%'}")

print("\n" + "=" * 50)

