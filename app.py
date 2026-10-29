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
# CUSTOM CSS (UI ONLY)
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --cacao: #5A3A22;
        --cacao-dark: #3E2716;
        --green: #4E7C46;
        --green-soft: #E8F1E4;
        --cream: #FBF7F0;
        --cream-2: #F5EEE2;
        --amber: #D99A2B;
        --amber-soft: #FCEFD2;
        --text: #2E241C;
        --muted: #7A6A5C;
        --border: #EFE4D3;
        --shadow: 0 10px 30px rgba(90, 58, 34, 0.10);
        --shadow-soft: 0 4px 14px rgba(90, 58, 34, 0.08);
    }

    /* Base */
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, #F3F8EE 0%, transparent 60%),
            radial-gradient(900px 500px at 110% 0%, #FCF3E3 0%, transparent 55%),
            var(--cream);
        color: var(--text);
    }
    html, body, [class*="css"] {
        font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    /* Hero */
    .cacao-hero {
        background: linear-gradient(135deg, #5A3A22 0%, #6E4A2B 55%, #4E7C46 130%);
        border-radius: 22px;
        padding: 34px 38px;
        color: #FFF9F0;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
        margin-bottom: 26px;
        animation: fadeIn 0.6s ease both;
    }
    .cacao-hero::after {
        content: "";
        position: absolute;
        right: -60px; top: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(217,154,43,0.55), transparent 70%);
        border-radius: 50%;
    }
    .cacao-hero h1 {
        margin: 0; font-size: 2.6rem;
        letter-spacing: 0.5px; font-weight: 800;
        color: #FFF9F0;
    }
    .cacao-hero p {
        margin: 10px 0 0 0;
        font-size: 1.05rem; color: #F6E9D6;
        max-width: 720px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.22);
        color: #FFF3DF;
        font-size: 0.78rem;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        padding: 5px 12px;
        border-radius: 999px;
        margin-bottom: 12px;
    }

    /* Workflow */
    .workflow {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 28px;
        animation: fadeIn 0.8s ease both;
    }
    .wf-step {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        box-shadow: var(--shadow-soft);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .wf-step:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 28px rgba(90, 58, 34, 0.14);
    }
    .wf-num {
        display: inline-flex;
        align-items: center; justify-content: center;
        width: 28px; height: 28px;
        border-radius: 50%;
        background: var(--amber-soft);
        color: #8A5A0E;
        font-weight: 700; font-size: 0.85rem;
        margin-bottom: 8px;
    }
    .wf-title { font-weight: 700; color: var(--cacao-dark); font-size: 0.98rem; }
    .wf-sub { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }

    /* Section heads */
    .section-head {
        margin: 36px 0 14px 0;
        animation: fadeIn 0.6s ease both;
    }
    .section-head h2 {
        margin: 0; font-size: 1.4rem;
        color: var(--cacao-dark); font-weight: 800;
    }
    .section-head p {
        margin: 6px 0 0 0;
        color: var(--muted); font-size: 0.92rem;
    }

    /* Generic card */
    .card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 22px 24px;
        box-shadow: var(--shadow-soft);
        transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;
        animation: fadeIn 0.6s ease both;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 34px rgba(90, 58, 34, 0.16);
        border-color: #E4D4BD;
    }
    .card-title {
        font-size: 0.82rem;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--green);
        font-weight: 700;
        margin-bottom: 4px;
    }
    .card-model { color: var(--muted); font-size: 0.8rem; margin-bottom: 14px; }
    .card-metric {
        font-size: 2.0rem; font-weight: 800;
        color: var(--cacao-dark); line-height: 1.1; margin-bottom: 4px;
    }
    .card-sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 12px; }

    /* Input panels (used for measurement groups) */
    .input-panel {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 22px 24px 10px 24px;
        box-shadow: var(--shadow-soft);
        animation: fadeIn 0.6s ease both;
        margin-bottom: 6px;
    }
    .input-panel-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .input-panel-title {
        font-size: 0.85rem;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--cacao-dark);
        font-weight: 800;
    }
    .input-panel-model {
        font-size: 0.78rem;
        color: var(--muted);
        background: var(--cream-2);
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
    }
    .input-panel-note {
        font-size: 0.85rem;
        color: var(--muted);
        margin-bottom: 14px;
    }

    /* Preview chips */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 4px 0 12px 0;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--green-soft);
        color: #3F6B37;
        border: 1px solid #D5E4CC;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .chip-amber {
        background: var(--amber-soft);
        color: #8A5A0E;
        border-color: #F1DEB0;
    }
    .chip-note {
        background: #F5EEE2;
        color: var(--muted);
        border-color: var(--border);
        font-weight: 500;
    }

    /* Progress */
    .progress-wrap {
        background: #F1E8DA;
        border-radius: 999px;
        height: 12px;
        width: 100%;
        overflow: hidden;
        margin: 8px 0 6px 0;
    }
    .progress-bar {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--green) 0%, #7FAF73 60%, var(--amber) 100%);
        width: 0;
        animation: growBar 1.1s cubic-bezier(0.22, 0.8, 0.28, 1) forwards;
    }
    .progress-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: var(--muted);
        margin-bottom: 2px;
    }

    /* Pills */
    .pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.4px;
    }
    .pill-low { background: #E8F1E4; color: #3F6B37; }
    .pill-medium { background: #FCEFD2; color: #8A5A0E; }
    .pill-severe { background: #F8E0D8; color: #A6412A; }

    /* Recommendation */
    .recommend-card {
        background: linear-gradient(135deg, #F4FAF0 0%, #FBF4E6 100%);
        border: 1px solid #E2EAD9;
        border-radius: 22px;
        padding: 28px 30px;
        box-shadow: var(--shadow);
        animation: fadeIn 0.7s ease both;
    }
    .recommend-card .label {
        font-size: 0.78rem;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: var(--green);
        font-weight: 700;
    }
    .recommend-card .value {
        font-size: 1.9rem;
        font-weight: 800;
        color: var(--cacao-dark);
        margin: 8px 0 6px 0;
    }
    .recommend-card .conf { color: var(--muted); font-size: 0.9rem; }

    /* Analyze action area */
    .analyze-wrap {
        margin: 28px 0 10px 0;
        padding: 22px;
        background: linear-gradient(135deg, #F5EEE2 0%, #FBF7F0 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        text-align: center;
        box-shadow: var(--shadow-soft);
    }
    .analyze-hint {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 10px;
    }

    /* Streamlit buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--cacao) 0%, #6E4A2B 100%);
        color: #FFF7EA;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 1.6rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        box-shadow: 0 8px 20px rgba(90, 58, 34, 0.22);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(90, 58, 34, 0.30);
        filter: brightness(1.05);
        color: #FFF7EA;
    }
    .stButton > button:active { transform: translateY(0); }

    /* Uploader */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1.5px dashed #D9C6A9;
        border-radius: 18px;
        padding: 14px;
        box-shadow: var(--shadow-soft);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--amber);
        box-shadow: 0 10px 24px rgba(217, 154, 43, 0.18);
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: none;
        margin-top: 8px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: var(--shadow-soft);
    }

    /* Inputs styling */
    .stNumberInput input, .stTextInput input {
        background: #FBF7F0 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] > div {
        background: #FBF7F0 !important;
        border-radius: 10px !important;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes growBar { from { width: 0; } }

    @media (max-width: 900px) {
        .workflow { grid-template-columns: repeat(2, 1fr); }
        .cacao-hero h1 { font-size: 2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
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
    return joblib.load("models/cocoa_ripeness_glcm_svm_54features.pkl")


@st.cache_resource
def load_ann():
    return tf.keras.models.load_model("models/ann_cacao_disease_model.keras")


@st.cache_resource
def load_linear_regression():
    return joblib.load("models/severity_regressor_model.joblib")


# ============================================================
# SVM FEATURE EXTRACTION
# ============================================================

def extract_svm_features(uploaded_file, size=(128, 128)):

    image_bytes = uploaded_file.getvalue()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not read the uploaded image.")

    image = cv2.resize(image, size)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    color_features = []
    for channel in cv2.split(hsv):
        color_features.extend([float(channel.mean()), float(channel.std())])

    gray_q = (gray // 16).astype(np.uint8)
    glcm = graycomatrix(
        gray_q,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True
    )

    texture_features = []
    for prop in ["contrast", "dissimilarity", "homogeneity",
                 "energy", "correlation", "ASM"]:
        texture_features.extend(graycoprops(glcm, prop).ravel().tolist())

    return np.asarray(color_features + texture_features, dtype=np.float32)


# ============================================================
# SVM PREDICTION
# ============================================================

def predict_svm_ripeness(uploaded_file, model_bundle):

    features = extract_svm_features(uploaded_file)
    fitted_model = model_bundle["model"]
    probabilities = fitted_model.predict_proba([features])[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = str(fitted_model.classes_[predicted_index])
    confidence = float(probabilities[predicted_index])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(fitted_model.classes_, probabilities)
        }
    }


# ============================================================
# ANN DISEASE PREDICTION
# ============================================================

def predict_ann_disease(uploaded_file, model):

    image_bytes = uploaded_file.getvalue()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not read the uploaded image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)

    probabilities = model.predict(image, verbose=0)[0]

    class_names = ["Healthy", "Black Pod", "Cacao Pod Borer"]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }
    }


# ============================================================
# LINEAR REGRESSION — SEVERITY
# ============================================================

def extract_severity_features(uploaded_file):

    image_bytes = uploaded_file.getvalue()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not read the uploaded image.")

    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    lesion_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    lesion_ratio = (np.sum(lesion_mask > 0) / (128 * 128)) * 100.0

    mean_val = float(np.mean(hsv[:, :, 2]))
    std_val = float(np.std(hsv[:, :, 2]))

    return [lesion_ratio, mean_val, std_val]


def predict_linear_regression(uploaded_file, model):

    features = extract_severity_features(uploaded_file)
    raw_severity = model.predict([features])[0]
    severity_percentage = float(np.clip(raw_severity, 0.0, 100.0))

    if severity_percentage <= 25.0:
        severity_level = "Low"
    elif severity_percentage <= 50.0:
        severity_level = "Medium"
    else:
        severity_level = "Severe"

    return {
        "severity_percentage": severity_percentage,
        "severity_level": severity_level,
        "features": {
            "lesion_ratio": float(features[0]),
            "mean_val": float(features[1]),
            "std_val": float(features[2])
        }
    }


# ============================================================
# NAIVE BAYES PREDICTION
# ============================================================

def predict_naive_bayes(model, moisture, cadmium, polyphenols):

    features = pd.DataFrame([{
        "moisture": moisture,
        "cadmium": cadmium,
        "polyphenols": polyphenols
    }])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_names = model.classes_

    predicted_index = list(class_names).index(prediction)
    confidence = float(probabilities[predicted_index])

    return {
        "predicted_class": str(prediction),
        "confidence": confidence,
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(class_names, probabilities)
        }
    }


# ============================================================
# DECISION TREE PREDICTION
# ============================================================

def predict_decision_tree(bundle, bean_size, ripeness, moisture_level, moisture_percent):

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    input_data = pd.DataFrame([{
        "Bean_Size": bean_size,
        "Ripeness": ripeness,
        "Moisture_Level": moisture_level,
        "Moisture_Percent": moisture_percent
    }])

    transformed_data = preprocessor.transform(input_data)
    prediction = model.predict(transformed_data)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_data)[0]
        confidence = float(np.max(probabilities))

    return {
        "predicted_class": str(prediction),
        "confidence": confidence
    }


# ============================================================
# LOAD MODELS
# ============================================================

try:
    nb_model = load_naive_bayes()
except Exception as e:
    nb_model = None
    st.error(f"Could not load Naive Bayes model: {e}")

try:
    dt_bundle = load_decision_tree()
except Exception as e:
    dt_bundle = None
    st.error(f"Could not load Decision Tree model: {e}")

try:
    svm_bundle = load_svm()
except Exception as e:
    svm_bundle = None
    st.error(f"Could not load SVM model: {e}")

try:
    ann_model = load_ann()
except Exception as e:
    ann_model = None
    st.error(f"Could not load ANN model: {e}")

try:
    lr_model = load_linear_regression()
except Exception as e:
    lr_model = None
    st.error(f"Could not load Linear Regression model: {e}")


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="cacao-hero">
        <div class="hero-badge">AI Agricultural Dashboard</div>
        <h1>🌱 Cacao AI</h1>
        <p>Intelligent Cacao Analysis for Better Harvest Decisions</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WORKFLOW STRIP
# ============================================================

st.markdown(
    """
    <div class="workflow">
        <div class="wf-step">
            <div class="wf-num">1</div>
            <div class="wf-title">Upload</div>
            <div class="wf-sub">Add a cacao pod image</div>
        </div>
        <div class="wf-step">
            <div class="wf-num">2</div>
            <div class="wf-title">Analyze</div>
            <div class="wf-sub">Run the AI models</div>
        </div>
        <div class="wf-step">
            <div class="wf-num">3</div>
            <div class="wf-title">Review Results</div>
            <div class="wf-sub">See predictions &amp; confidence</div>
        </div>
        <div class="wf-step">
            <div class="wf-num">4</div>
            <div class="wf-title">Get Recommendation</div>
            <div class="wf-sub">Post-harvest guidance</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 1. UPLOAD CACAO IMAGE
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <h2>1. Upload Cacao Image</h2>
        <p>The uploaded image is the primary input for the ripeness, disease, and severity models.</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a cacao image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col_img, col_info = st.columns([1.1, 1], gap="large")

    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(uploaded_file, caption="Uploaded Cacao Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Image Ready</div>
                <div class="card-model">Your image has been loaded successfully.</div>
                <div class="card-sub">
                    Click <strong>Analyze Cacao</strong> to run the ripeness,
                    disease detection, and disease severity models. You can
                    optionally fill in the measurements below before analyzing.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 2. ADDITIONAL MEASUREMENTS (NAIVE BAYES)
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <h2>Additional Measurements</h2>
        <p>
            These are laboratory / post-harvest measurements used by the
            fermentation quality model. They are not derived from the uploaded image.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="input-panel">
        <div class="input-panel-head">
            <div class="input-panel-title">Fermentation Quality Inputs</div>
            <div class="input-panel-model">Naive Bayes</div>
        </div>
        <div class="input-panel-note">
            Enter the measured moisture, cadmium, and polyphenol values.
        </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
with col1:
    moisture = st.number_input("Moisture", min_value=0.0, value=5.0, step=0.1)
with col2:
    cadmium = st.number_input("Cadmium", min_value=0.0, value=2.0, step=0.1)
with col3:
    polyphenols = st.number_input("Polyphenols", min_value=0.0, value=30.0, step=0.1)

st.markdown(
    f"""
        <div class="chip-row">
            <span class="chip">💧 Moisture: {moisture:.2f}</span>
            <span class="chip chip-amber">🧪 Cadmium: {cadmium:.2f}</span>
            <span class="chip">🍫 Polyphenols: {polyphenols:.2f}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. POST-HARVEST INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <h2>Post-Harvest Information</h2>
        <p>
            Additional information used by the post-harvest recommendation model.
            Ripeness is taken automatically from the SVM prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="input-panel">
        <div class="input-panel-head">
            <div class="input-panel-title">Recommendation Inputs</div>
            <div class="input-panel-model">Decision Tree</div>
        </div>
        <div class="input-panel-note">
            Provide the bean size, moisture level, and moisture percentage.
        </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
with col1:
    bean_size = st.selectbox("Bean Size", ["Small", "Medium", "Large"])
with col2:
    moisture_level = st.selectbox("Moisture Level", ["Low", "Medium", "High"])

moisture_percent = st.number_input(
    "Moisture Percentage",
    min_value=0.0,
    max_value=100.0,
    value=7.0,
    step=0.1
)

st.markdown(
    """
        <div class="chip-row">
            <span class="chip">🫘 Bean Size: {bs}</span>
            <span class="chip chip-amber">💧 Moisture Level: {ml}</span>
            <span class="chip">📊 Moisture %: {mp:.2f}</span>
            <span class="chip chip-note">🌱 Ripeness: from SVM</span>
        </div>
    </div>
    """.format(bs=bean_size, ml=moisture_level, mp=moisture_percent),
    unsafe_allow_html=True
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown(
    """
    <div class="analyze-wrap">
        <div class="analyze-hint">Ready when you are — run all models in one click.</div>
    </div>
    """,
    unsafe_allow_html=True
)

analyze = st.button("🔍 Analyze Cacao", type="primary")


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if uploaded_file is None:
        st.warning("Please upload a cacao image first.")
        st.stop()

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-head">
            <h2>3. Review Results</h2>
            <p>AI predictions from the ripeness, disease detection, and severity models.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- SVM ----
    svm_result = None
    if svm_bundle is not None:
        try:
            svm_result = predict_svm_ripeness(uploaded_file, svm_bundle)
        except Exception as e:
            st.error(f"SVM prediction failed: {e}")

    # ---- ANN ----
    ann_result = None
    if ann_model is not None:
        try:
            ann_result = predict_ann_disease(uploaded_file, ann_model)
        except Exception as e:
            st.error(f"ANN prediction failed: {e}")

    # ---- Linear Regression ----
    lr_result = None
    if lr_model is not None:
        try:
            lr_result = predict_linear_regression(uploaded_file, lr_model)
        except Exception as e:
            st.error(f"Linear Regression prediction failed: {e}")

    # ---- Result cards ----
    r1, r2, r3 = st.columns(3, gap="large")

    with r1:
        if svm_result is not None:
            conf_pct = svm_result["confidence"] * 100.0
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">AI Ripeness Assessment</div>
                    <div class="card-model">Support Vector Machine (SVM)</div>
                    <div class="card-metric">{svm_result['predicted_class']}</div>
                    <div class="card-sub">Predicted ripeness class</div>
                    <div class="progress-label">
                        <span>Confidence</span><span>{conf_pct:.2f}%</span>
                    </div>
                    <div class="progress-wrap">
                        <div class="progress-bar" style="width: {conf_pct:.2f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander("Probability Breakdown"):
                for label, probability in svm_result["probabilities"].items():
                    pct = probability * 100.0
                    st.markdown(
                        f"""
                        <div class="progress-label"><span>{label}</span><span>{pct:.2f}%</span></div>
                        <div class="progress-wrap"><div class="progress-bar" style="width: {pct:.2f}%;"></div></div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                """
                <div class="card">
                    <div class="card-title">AI Ripeness Assessment</div>
                    <div class="card-model">Support Vector Machine (SVM)</div>
                    <div class="card-sub">Model not available.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with r2:
        if ann_result is not None:
            conf_pct = ann_result["confidence"] * 100.0
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">AI Disease Detection</div>
                    <div class="card-model">Artificial Neural Network (ANN)</div>
                    <div class="card-metric">{ann_result['predicted_class']}</div>
                    <div class="card-sub">Detected condition</div>
                    <div class="progress-label">
                        <span>Confidence</span><span>{conf_pct:.2f}%</span>
                    </div>
                    <div class="progress-wrap">
                        <div class="progress-bar" style="width: {conf_pct:.2f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander("Probability Breakdown"):
                for label, probability in ann_result["probabilities"].items():
                    pct = probability * 100.0
                    st.markdown(
                        f"""
                        <div class="progress-label"><span>{label}</span><span>{pct:.2f}%</span></div>
                        <div class="progress-wrap"><div class="progress-bar" style="width: {pct:.2f}%;"></div></div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                """
                <div class="card">
                    <div class="card-title">AI Disease Detection</div>
                    <div class="card-model">Artificial Neural Network (ANN)</div>
                    <div class="card-sub">Model not available.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with r3:
        if lr_result is not None:
            sev = lr_result["severity_percentage"]
            level = lr_result["severity_level"]
            pill_class = {
                "Low": "pill-low",
                "Medium": "pill-medium",
                "Severe": "pill-severe"
            }.get(level, "pill-low")

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">AI Disease Severity</div>
                    <div class="card-model">Linear Regression</div>
                    <div class="card-metric">{sev:.2f}%</div>
                    <div class="card-sub">Estimated severity level</div>
                    <div class="progress-label">
                        <span>Severity</span><span>{sev:.2f}%</span>
                    </div>
                    <div class="progress-wrap">
                        <div class="progress-bar" style="width: {sev:.2f}%;"></div>
                    </div>
                    <div style="margin-top: 12px;">
                        <span class="pill {pill_class}">{level}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander("Technical Image Analysis"):
                st.markdown(
                    f"""
                    <div class="progress-label">
                        <span>Lesion Ratio</span><span>{lr_result['features']['lesion_ratio']:.2f}%</span>
                    </div>
                    <div class="progress-label">
                        <span>Mean Brightness</span><span>{lr_result['features']['mean_val']:.2f}</span>
                    </div>
                    <div class="progress-label">
                        <span>Brightness Variation</span><span>{lr_result['features']['std_val']:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div class="card">
                    <div class="card-title">AI Disease Severity</div>
                    <div class="card-model">Linear Regression</div>
                    <div class="card-sub">Model not available.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ---- Naive Bayes ----
    if nb_model is not None:
        try:
            nb_result = predict_naive_bayes(nb_model, moisture, cadmium, polyphenols)
            conf_pct = nb_result["confidence"] * 100.0

            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Fermentation Quality</div>
                    <div class="card-model">Naive Bayes</div>
                    <div class="card-metric">{nb_result['predicted_class']}</div>
                    <div class="card-sub">Based on laboratory measurements</div>
                    <div class="progress-label">
                        <span>Confidence</span><span>{conf_pct:.2f}%</span>
                    </div>
                    <div class="progress-wrap">
                        <div class="progress-bar" style="width: {conf_pct:.2f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander("Probability Breakdown"):
                for label, probability in nb_result["probabilities"].items():
                    pct = probability * 100.0
                    st.markdown(
                        f"""
                        <div class="progress-label"><span>{label}</span><span>{pct:.2f}%</span></div>
                        <div class="progress-wrap"><div class="progress-bar" style="width: {pct:.2f}%;"></div></div>
                        """,
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"Naive Bayes prediction failed: {e}")

    # ---- Decision Tree ----
    if dt_bundle is not None:
        try:
            if svm_result is not None:
                predicted_ripeness = svm_result["predicted_class"]
                ripeness_mapping = {
                    "Immature": "Unripe",
                    "Mature": "Optimal Harvest",
                    "Overmature": "Overripe",
                    "Unripe": "Unripe",
                    "Ripe": "Optimal Harvest",
                    "Overripe": "Overripe"
                }
                decision_tree_ripeness = ripeness_mapping.get(
                    predicted_ripeness, "Optimal Harvest"
                )
            else:
                decision_tree_ripeness = "Optimal Harvest"

            dt_result = predict_decision_tree(
                dt_bundle,
                bean_size,
                decision_tree_ripeness,
                moisture_level,
                moisture_percent
            )

            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="section-head">
                    <h2>4. Get Recommendation</h2>
                    <p>Post-harvest recommendation from the Decision Tree model.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            conf_line = ""
            if dt_result["confidence"] is not None:
                conf_line = (
                    f'<div class="conf">Confidence: '
                    f'{dt_result["confidence"] * 100:.2f}%</div>'
                )

            st.markdown(
                f"""
                <div class="recommend-card">
                    <div class="label">Decision Tree · Post-Harvest Recommendation</div>
                    <div class="value">{dt_result['predicted_class']}</div>
                    {conf_line}
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Decision Tree prediction failed: {e}")

    # ---- Model status ----
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-head">
            <h2>Model Integration Status</h2>
            <p>Overview of the machine learning models used in this app.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    status_data = pd.DataFrame({
        "Algorithm": [
            "Naive Bayes", "Decision Tree", "ANN", "SVM",
            "KNN", "Linear Regression"
        ],
        "Purpose": [
            "Fermentation / Quality",
            "Post-Harvest Recommendation",
            "Disease Detection",
            "Cacao Ripeness",
            "Cacao Ripeness / Classification",
            "Disease Severity"
        ],
        "Status": [
            "Integrated", "Integrated", "Integrated", "Integrated",
            "Temporarily pending feature extraction", "Integrated"
        ]
    })

    st.dataframe(status_data, use_container_width=True, hide_index=True)