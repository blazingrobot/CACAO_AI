import streamlit as st
import joblib
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

from skimage.feature import graycomatrix, graycoprops


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cacao AI",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_naive_bayes():
    return joblib.load("models/cocoa_naive_bayes_model.pkl")


@st.cache_resource
def load_decision_tree():
    return joblib.load("models/cocoa_decision_tree_model.pkl")


@st.cache_resource
def load_svm():
    return joblib.load(
        "models/cocoa_ripeness_glcm_svm_54features.pkl"
    )


@st.cache_resource
def load_ann():
    return tf.keras.models.load_model(
        "models/ann_cacao_disease_model.keras"
    )


# ============================================================
# SVM FEATURE EXTRACTION
# ============================================================

def extract_svm_features(uploaded_file, size=(128, 128)):

    image_bytes = uploaded_file.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not read the uploaded image."
        )

    image = cv2.resize(
        image,
        size
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # HSV COLOR FEATURES
    # --------------------------------------------------------

    color_features = []

    for channel in cv2.split(hsv):
        color_features.extend(
            [
                float(channel.mean()),
                float(channel.std())
            ]
        )

    # --------------------------------------------------------
    # GLCM TEXTURE FEATURES
    # --------------------------------------------------------

    gray_q = (
        gray // 16
    ).astype(np.uint8)

    glcm = graycomatrix(
        gray_q,
        distances=[1, 2],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=16,
        symmetric=True,
        normed=True
    )

    texture_features = []

    for prop in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]:
        texture_features.extend(
            graycoprops(
                glcm,
                prop
            ).ravel().tolist()
        )

    features = np.asarray(
        color_features + texture_features,
        dtype=np.float32
    )

    return features


# ============================================================
# SVM PREDICTION
# ============================================================

def predict_svm_ripeness(
    uploaded_file,
    model_bundle
):

    features = extract_svm_features(
        uploaded_file
    )

    fitted_model = model_bundle["model"]

    probabilities = fitted_model.predict_proba(
        [features]
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = str(
        fitted_model.classes_[predicted_index]
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            str(label): float(probability)
            for label, probability
            in zip(
                fitted_model.classes_,
                probabilities
            )
        }
    }


# ============================================================
# ANN DISEASE PREDICTION
# ============================================================

def predict_ann_disease(
    uploaded_file,
    model
):

    image_bytes = uploaded_file.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not read the uploaded image."
        )

    # Convert OpenCV BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ANN expects 128 x 128
    image = cv2.resize(
        image,
        (128, 128)
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # IMPORTANT:
    # The model already contains its own Rescaling layer.
    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    class_names = [
        "Healthy",
        "Black Pod",
        "Cacao Pod Borer"
    ]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            class_names[i]: float(
                probabilities[i]
            )
            for i in range(
                len(class_names)
            )
        }
    }


# ============================================================
# NAIVE BAYES PREDICTION
# ============================================================

def predict_naive_bayes(
    model,
    moisture,
    cadmium,
    polyphenols
):

    features = pd.DataFrame(
        [
            {
                "moisture": moisture,
                "cadmium": cadmium,
                "polyphenols": polyphenols
            }
        ]
    )

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    class_names = model.classes_

    predicted_index = list(
        class_names
    ).index(prediction)

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "predicted_class": str(
            prediction
        ),
        "confidence": confidence,
        "probabilities": {
            str(label): float(probability)
            for label, probability
            in zip(
                class_names,
                probabilities
            )
        }
    }


# ============================================================
# DECISION TREE PREDICTION
# ============================================================

def predict_decision_tree(
    bundle,
    bean_size,
    ripeness,
    moisture_level,
    moisture_percent
):

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    input_data = pd.DataFrame(
        [
            {
                "Bean_Size": bean_size,
                "Ripeness": ripeness,
                "Moisture_Level": moisture_level,
                "Moisture_Percent": moisture_percent
            }
        ]
    )

    transformed_data = preprocessor.transform(
        input_data
    )

    prediction = model.predict(
        transformed_data
    )[0]

    confidence = None

    if hasattr(
        model,
        "predict_proba"
    ):
        probabilities = model.predict_proba(
            transformed_data
        )[0]

        confidence = float(
            np.max(probabilities)
        )

    return {
        "predicted_class": str(
            prediction
        ),
        "confidence": confidence
    }


# ============================================================
# LOAD MODELS
# ============================================================

try:

    nb_model = load_naive_bayes()

except Exception as e:

    nb_model = None
    st.error(
        f"Could not load Naive Bayes model: {e}"
    )


try:

    dt_bundle = load_decision_tree()

except Exception as e:

    dt_bundle = None
    st.error(
        f"Could not load Decision Tree model: {e}"
    )


try:

    svm_bundle = load_svm()

except Exception as e:

    svm_bundle = None
    st.error(
        f"Could not load SVM model: {e}"
    )


try:

    ann_model = load_ann()

except Exception as e:

    ann_model = None
    st.error(
        f"Could not load ANN model: {e}"
    )


# ============================================================
# TITLE
# ============================================================

st.title("Cacao AI")

st.write(
    "AI-based cacao analysis using multiple "
    "machine learning models."
)


# ============================================================
# 1. UPLOAD CACAO IMAGE
# ============================================================

st.header("1. Upload Cacao Image")

uploaded_file = st.file_uploader(
    "Upload a cacao image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Cacao Image",
        width=400
    )


# ============================================================
# 2. CACAO MEASUREMENTS
# ============================================================

st.header("2. Cacao Measurements")

st.write(
    "Enter the cacao measurements required "
    "for fermentation and quality analysis."
)

col1, col2, col3 = st.columns(3)

with col1:

    moisture = st.number_input(
        "Moisture",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

with col2:

    cadmium = st.number_input(
        "Cadmium",
        min_value=0.0,
        value=2.0,
        step=0.1
    )

with col3:

    polyphenols = st.number_input(
        "Polyphenols",
        min_value=0.0,
        value=30.0,
        step=0.1
    )


# ============================================================
# 3. POST-HARVEST INFORMATION
# ============================================================

st.header("3. Post-Harvest Information")

col1, col2 = st.columns(2)

with col1:

    bean_size = st.selectbox(
        "Bean Size",
        [
            "Small",
            "Medium",
            "Large"
        ]
    )

    ripeness = st.selectbox(
        "Ripeness",
        [
            "Unripe",
            "Optimal Harvest",
            "Overripe"
        ]
    )

with col2:

    moisture_level = st.selectbox(
        "Moisture Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    moisture_percent = st.number_input(
        "Moisture Percentage",
        min_value=0.0,
        max_value=100.0,
        value=7.0,
        step=0.1
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "Analyze Cacao",
    type="primary"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if uploaded_file is None:

        st.warning(
            "Please upload a cacao image first."
        )

        st.stop()

    st.divider()

    st.header("Analysis Results")


    # ========================================================
    # SVM — RIPENESS
    # ========================================================

    if svm_bundle is not None:

        try:

            svm_result = predict_svm_ripeness(
                uploaded_file,
                svm_bundle
            )

            st.subheader(
                "SVM — Cacao Ripeness"
            )

            st.write(
                f"**Predicted Ripeness:** "
                f"{svm_result['predicted_class']}"
            )

            st.write(
                f"**Confidence:** "
                f"{svm_result['confidence'] * 100:.2f}%"
            )

            st.write(
                "Probabilities:"
            )

            for label, probability in svm_result[
                "probabilities"
            ].items():

                st.write(
                    f"- {label}: "
                    f"{probability * 100:.2f}%"
                )

        except Exception as e:

            st.error(
                f"SVM prediction failed: {e}"
            )


    # ========================================================
    # ANN — DISEASE DETECTION
    # ========================================================

    if ann_model is not None:

        try:

            ann_result = predict_ann_disease(
                uploaded_file,
                ann_model
            )

            st.subheader(
                "ANN — Disease Detection"
            )

            st.write(
                f"**Predicted Disease:** "
                f"{ann_result['predicted_class']}"
            )

            st.write(
                f"**Confidence:** "
                f"{ann_result['confidence'] * 100:.2f}%"
            )

            st.write(
                "Probabilities:"
            )

            for label, probability in ann_result[
                "probabilities"
            ].items():

                st.write(
                    f"- {label}: "
                    f"{probability * 100:.2f}%"
                )

        except Exception as e:

            st.error(
                f"ANN prediction failed: {e}"
            )


    # ========================================================
    # NAIVE BAYES — FERMENTATION
    # ========================================================

    if nb_model is not None:

        try:

            nb_result = predict_naive_bayes(
                nb_model,
                moisture,
                cadmium,
                polyphenols
            )

            st.subheader(
                "Naive Bayes — Fermentation Quality"
            )

            st.write(
                f"**Prediction:** "
                f"{nb_result['predicted_class']}"
            )

            st.write(
                f"**Confidence:** "
                f"{nb_result['confidence'] * 100:.2f}%"
            )

            st.write(
                "Probabilities:"
            )

            for label, probability in nb_result[
                "probabilities"
            ].items():

                st.write(
                    f"- {label}: "
                    f"{probability * 100:.2f}%"
                )

        except Exception as e:

            st.error(
                f"Naive Bayes prediction failed: {e}"
            )


    # ========================================================
    # DECISION TREE — RECOMMENDATION
    # ========================================================

    if dt_bundle is not None:

        try:

            dt_result = predict_decision_tree(
                dt_bundle,
                bean_size,
                ripeness,
                moisture_level,
                moisture_percent
            )

            st.subheader(
                "Decision Tree — Post-Harvest Recommendation"
            )

            st.write(
                f"**Recommendation:** "
                f"{dt_result['predicted_class']}"
            )

            if dt_result["confidence"] is not None:

                st.write(
                    f"**Confidence:** "
                    f"{dt_result['confidence'] * 100:.2f}%"
                )

        except Exception as e:

            st.error(
                f"Decision Tree prediction failed: {e}"
            )


    # ========================================================
    # CURRENT MODEL STATUS
    # ========================================================

    st.divider()

    st.subheader(
        "Model Integration Status"
    )

    status_data = pd.DataFrame(
        {
            "Algorithm": [
                "Naive Bayes",
                "Decision Tree",
                "ANN",
                "SVM",
                "KNN",
                "Linear Regression"
            ],
            "Purpose": [
                "Fermentation / Quality",
                "Post-Harvest Recommendation",
                "Disease Detection",
                "Cacao Ripeness",
                "Pending Retraining",
                "Pending Feature Details"
            ],
            "Status": [
                "Integrated",
                "Integrated",
                "Integrated",
                "Integrated",
                "Needs retraining",
                "Needs feature details"
            ]
        }
    )

    st.dataframe(
        status_data,
        use_container_width=True,
        hide_index=True
    )