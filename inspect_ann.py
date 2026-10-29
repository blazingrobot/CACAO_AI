import tensorflow as tf

model_path = "models/ann_cacao_disease_model.keras"

print("Loading ANN model...")

model = tf.keras.models.load_model(model_path)

print("\n==============================")
print("ANN MODEL INFORMATION")
print("==============================")

print("\nModel:")
model.summary()

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\nInput dtype:")
print(model.inputs[0].dtype)

print("\nOutput dtype:")
print(model.outputs[0].dtype)

print("\n==============================")
print("DONE")
print("==============================")