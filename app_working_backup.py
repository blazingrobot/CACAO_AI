import streamlit as st
import joblib
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cacao AI",
    page_icon="🍫",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_naive_bayes():
    return joblib.load("models/cocoa_naive_bayes_model.pkl")


@st.cache_resource
def load_decision_tree():
    data = joblib.load("models/cocoa_decision_tree_model.pkl")

    return (
        data["model"],
        data["preprocessor"],
        data["feature_columns"]
    )


# ============================================================
# LOAD THE TRAINED MODELS
# ============================================================

try:
    naive_bayes_model = load_naive_bayes()

    decision_tree_model, decision_tree_preprocessor, decision_tree_features = (
        load_decision_tree()
    )

    models_loaded = True

except Exception as error:
    models_loaded = False
    model_error = error


# ============================================================
# HEADER
# ============================================================

st.title("Cacao AI")
st.subheader("Cacao Quality & Health Analysis")

st.write(
    "Upload a cacao image and provide the available cacao measurements "
    "to analyze fermentation condition and post-harvest recommendations."
)

st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("1. Upload Cacao Image")

uploaded_file = st.file_uploader(
    "Choose a cacao image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Cacao Image",
        width=450
    )

    st.success("Cacao image uploaded successfully!")


st.divider()


# ============================================================
# CACAO MEASUREMENTS
# ============================================================

st.header("2. Cacao Measurements")

st.write(
    "Enter the available measurements used by the trained "
    "machine learning models."
)

col1, col2, col3 = st.columns(3)

with col1:
    moisture = st.number_input(
        "Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.01
    )

with col2:
    cadmium = st.number_input(
        "Cadmium",
        min_value=0.0,
        value=2.0,
        step=0.01
    )

with col3:
    polyphenols = st.number_input(
        "Polyphenols",
        min_value=0.0,
        value=30.0,
        step=0.01
    )


st.divider()


# ============================================================
# DECISION TREE INPUTS
# ============================================================

st.subheader("Post-Harvest Information")

col4, col5 = st.columns(2)

with col4:

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


with col5:

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
        value=5.0,
        step=0.01
    )


st.divider()


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("Analyze Cacao", type="primary", use_container_width=True):

    if uploaded_file is None:

        st.warning("Please upload a cacao image first.")

    elif not models_loaded:

        st.error("Unable to load the trained models.")

        st.code(str(model_error))

    else:

        # ====================================================
        # NAIVE BAYES PREDICTION
        # ====================================================

        naive_bayes_input = pd.DataFrame(
            [[
                moisture,
                cadmium,
                polyphenols
            ]],
            columns=[
                "moisture",
                "cadmium",
                "polyphenols"
            ]
        )

        nb_prediction = naive_bayes_model.predict(
            naive_bayes_input
        )[0]

        # Probability
        nb_probabilities = naive_bayes_model.predict_proba(
            naive_bayes_input
        )[0]

        nb_classes = naive_bayes_model.classes_

        nb_confidence = max(nb_probabilities) * 100


        # ====================================================
        # DECISION TREE PREDICTION
        # ====================================================

        decision_tree_input = pd.DataFrame(
            [[
                bean_size,
                ripeness,
                moisture_level,
                moisture_percent
            ]],
            columns=decision_tree_features
        )

        processed_input = decision_tree_preprocessor.transform(
            decision_tree_input
        )

        dt_prediction = decision_tree_model.predict(
            processed_input
        )[0]


        # ====================================================
        # RESULTS
        # ====================================================

        st.header("3. Analysis Results")

        st.success("Analysis completed successfully.")

        st.divider()


        # ====================================================
        # NAIVE BAYES RESULT
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Fermentation / Quality")

            st.caption("Algorithm: Naive Bayes")

            st.metric(
                label="Predicted Condition",
                value=nb_prediction
            )

            st.write(
                f"Model confidence: **{nb_confidence:.2f}%**"
            )


        # ====================================================
        # DECISION TREE RESULT
        # ====================================================

        with col2:

            st.subheader("Post-Harvest Recommendation")

            st.caption("Algorithm: Decision Tree")

            st.info(dt_prediction)


        st.divider()


        # ====================================================
        # OTHER MODELS
        # ====================================================

        st.header("Other Machine Learning Analysis")

        st.caption(
            "These models are part of the proposed system and "
            "will be connected when their trained models are available."
        )

        col3, col4, col5, col6 = st.columns(4)

        with col3:
            st.subheader("Disease Detection")
            st.caption("ANN")
            st.warning("Model not connected yet.")

        with col4:
            st.subheader("Bean Quality")
            st.caption("SVM")
            st.warning("Model not available yet.")

        with col5:
            st.subheader("Pod Ripeness")
            st.caption("KNN")
            st.warning("Model not connected yet.")

        with col6:
            st.subheader("Infection Severity")
            st.caption("Linear Regression")
            st.warning("Model not connected yet.")


        st.divider()


        # ====================================================
        # OVERALL ASSESSMENT
        # ====================================================

        st.header("Overall Assessment")

        st.write(
            f"The Naive Bayes model classified the cacao sample as "
            f"**{nb_prediction}**."
        )

        st.write(
            f"The Decision Tree recommends: **{dt_prediction}**."
        )

        st.info(
            "Additional disease, bean quality, ripeness, and severity "
            "analysis will be available once the remaining trained "
            "models are connected."
        )