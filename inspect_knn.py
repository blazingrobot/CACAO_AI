import joblib
import numpy as np

PATH = "models/knn_cacao_model-1.pkl"

model = joblib.load(PATH)

print("=" * 60)
print("KNN MODEL INSPECTION")
print("=" * 60)

print("TYPE:")
print(type(model))

print("\nMODEL:")
print(model)

if isinstance(model, dict):

    print("\nDICTIONARY KEYS:")
    print(model.keys())

    for key, value in model.items():
        print(f"\n--- {key} ---")

        if hasattr(value, "shape"):
            print("TYPE:", type(value))
            print("SHAPE:", value.shape)

        elif isinstance(value, np.ndarray):
            print("TYPE:", type(value))
            print("SHAPE:", value.shape)
            print("VALUES:", value[:20])

        else:
            print("TYPE:", type(value))
            print("VALUE:", value)

print("\n" + "=" * 60)

if hasattr(model, "classes_"):
    print("CLASSES:")
    print(model.classes_)

if hasattr(model, "n_features_in_"):
    print("\nNUMBER OF FEATURES:")
    print(model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("\nFEATURE NAMES:")
    print(model.feature_names_in_)

if hasattr(model, "_fit_X"):
    print("\nTRAINING X:")
    print("Shape:", model._fit_X.shape)
    print("First row:", model._fit_X[0])

if hasattr(model, "_y"):
    print("\nTRAINING Y:")
    print("Shape:", model._y.shape)
    print("Unique labels:", np.unique(
        model._y,
        return_counts=True
    ))

print("=" * 60)