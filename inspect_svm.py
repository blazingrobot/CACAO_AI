import joblib

model_path = "models/cocoa_knn_model.pkl"
scaler_path = "models/cocoa_scaler.pkl"

print("Loading SVM model...")
model = joblib.load(model_path)

print("Loading scaler...")
scaler = joblib.load(scaler_path)

print("\n==============================")
print("SVM MODEL INFORMATION")
print("==============================")

print("\nModel type:")
print(type(model))

print("\nModel:")
print(model)

print("\nModel attributes:")

for attribute in [
    "classes_",
    "feature_names_in_",
    "n_features_in_"
]:
    if hasattr(model, attribute):
        print(f"\n{attribute}:")
        print(getattr(model, attribute))

print("\n==============================")
print("SCALER INFORMATION")
print("==============================")

print("\nScaler type:")
print(type(scaler))

print("\nScaler:")
print(scaler)

for attribute in [
    "feature_names_in_",
    "n_features_in_",
    "mean_",
    "scale_"
]:
    if hasattr(scaler, attribute):
        print(f"\n{attribute}:")
        print(getattr(scaler, attribute))

print("\n==============================")
print("DONE")
print("==============================")