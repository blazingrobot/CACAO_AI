import joblib
import numpy as np

MODEL_PATH = "models/knn_cacao_model.pkl"

model = joblib.load(MODEL_PATH)

print("=" * 60)
print("KNN PREDICTION TEST")
print("=" * 60)

print("Model:", model)
print("Classes:", model.classes_)
print("Features:", model.n_features_in_)

# Use the first training sample as a test input
sample = model._fit_X[0].reshape(1, -1)

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("\nTest Input:")
print(sample)

print("\nPrediction:")
print(prediction[0])

print("\nPredicted Class:")
print(prediction[0])

print("\nProbabilities:")
for class_name, prob in zip(model.classes_, probability[0]):
    print(f"{class_name}: {prob * 100:.2f}%")

print("=" * 60)