import joblib
import pandas as pd

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model_path = "models/cocoa_naive_bayes_model.pkl"

print("Loading Naive Bayes model...")

model = joblib.load(model_path)

print("Model loaded successfully!")

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

dataset_path = "datasets/cocoa_fermentation_training_dataset.csv"

df = pd.read_csv(dataset_path)

# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

features = [
    "moisture",
    "cadmium",
    "polyphenols"
]

X = df[features]

# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n==============================")
print("NAIVE BAYES TEST RESULTS")
print("==============================")

for index in range(len(df)):
    actual = df.iloc[index]["fermentation_class"]
    predicted = predictions[index]

    print(
        f"Row {index + 1}: "
        f"Actual = {actual} | "
        f"Predicted = {predicted}"
    )

# --------------------------------------------------
# ACCURACY
# --------------------------------------------------

correct = (predictions == df["fermentation_class"]).sum()
total = len(df)

accuracy = correct / total * 100

print("\n==============================")
print(f"Accuracy on dataset: {accuracy:.2f}%")
print("==============================")