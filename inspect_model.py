import pickle
import joblib

model_path = "models/cocoa_naive_bayes_model.pkl"
print("Loading model...")

try:
    model = joblib.load(model_path)
except Exception:
    with open(model_path, "rb") as file:
        model = pickle.load(file)

print("\n==============================")
print("MODEL INFORMATION")
print("==============================")

print("Model type:")
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
        print(f"{attribute}:")
        print(getattr(model, attribute))

print("\n==============================")
print("DONE")
print("==============================")