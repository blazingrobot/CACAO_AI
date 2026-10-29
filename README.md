# 🌱 Cacao AI

### Intelligent Cacao Analysis for Better Harvest Decisions

Cacao AI is a machine-learning-based web application designed to assist cacao farmers in analyzing cacao pods and making better post-harvest decisions.

The system allows a user to upload a cacao image and receive AI-assisted analysis for **ripeness, disease detection, and disease severity**. Additional measurements can also be provided for **fermentation quality** and **post-harvest recommendations**.

The application is built using **Python and Streamlit** and integrates multiple machine-learning algorithms.

---

# 📌 Table of Contents

1. [System Overview](#-system-overview)
2. [Requirements](#-requirements)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Option 1 — Download ZIP](#option-1--download-zip)
6. [Option 2 — Clone from GitHub](#option-2--clone-from-github)
7. [Check Python Version](#-check-python-version)
8. [Create a Virtual Environment](#-create-a-virtual-environment)
9. [Install Dependencies](#-install-dependencies)
10. [Run Cacao AI](#-run-cacao-ai)
11. [Using the Website](#-using-the-website)
12. [Machine Learning Models](#-machine-learning-models)
13. [System Workflow](#-system-workflow)
14. [Features](#-features)
15. [Current Integration Status](#-current-integration-status)
16. [Important Notes](#-important-notes)
17. [Stopping the Application](#-stopping-the-application)
18. [Project Purpose](#-project-purpose)

---

# 🌱 System Overview

Cacao AI is a web-based machine learning application intended to help cacao farmers analyze cacao samples using artificial intelligence.

The main input of the system is a **cacao image**.

From the uploaded image, the system can currently perform:

- Cacao ripeness classification
- Cacao disease detection
- Disease severity estimation

The system also supports additional measurements for:

- Fermentation quality
- Post-harvest recommendations

The application is designed to combine image-based analysis with additional cacao measurements to provide more useful information to the user.

---

# 💻 Requirements

Before running Cacao AI, make sure the computer has the following:

### Required

- Windows, macOS, or Linux
- Python
- Git (only required if cloning from GitHub)
- VS Code or another code editor
- Internet connection for downloading Python packages

### Recommended Python Version

The project was developed and tested using:

```text
Python 3.12.10
```

It is recommended to use **Python 3.12.x**.

Before installing anything, check the installed Python version.

Open **Command Prompt** or **PowerShell** and type:

```bash
python --version
```

Expected output should be similar to:

```text
Python 3.12.10
```

If `python` does not work, try:

```bash
py --version
```

---

# 📁 Project Structure

After downloading or cloning the project, the folder should contain approximately:

```text
CacaoAI/
│
├── datasets/
│   └── cocoa_fermentation_training_dataset.csv
│
├── models/
│   ├── ann_cacao_disease_model.keras
│   ├── cocoa_decision_tree_model.pkl
│   ├── cocoa_naive_bayes_model.pkl
│   ├── cocoa_ripeness_glcm_svm_54features.pkl
│   ├── knn_cacao_model.pkl
│   └── severity_regressor_model.joblib
│
├── .gitignore
├── app.py
├── app_working_backup.py
│
├── Cacao_Bean_SVM_GLCM_Kaggle.ipynb
├── Linear_Regression.ipynb
│
├── inspect_ann.py
├── inspect_dataset.py
├── inspect_decision_tree.py
├── inspect_knn.py
├── inspect_model.py
├── inspect_svm.py
├── test_knn.py
└── test_naive_bayes.py
```

The most important files for running the website are:

```text
app.py
models/
datasets/
```

The inspection and testing Python files are mainly for development and model checking.

---

# 📥 Installation

There are two ways to get the project.

You can either:

1. Download the project as a ZIP file from GitHub, or
2. Clone the repository using Git.

---

# Option 1 — Download ZIP

### Step 1 — Open the GitHub repository

Go to the Cacao AI GitHub repository.

### Step 2 — Download the repository

On GitHub:

1. Click the green **Code** button.
2. Click **Download ZIP**.
3. Wait for the ZIP file to download.

### Step 3 — Extract the ZIP

Find the downloaded ZIP file.

Right-click it and select:

**Extract All**

Extract it somewhere easy to access, such as:

```text
Downloads
```

or:

```text
Desktop
```

### Step 4 — Open the project folder

Open the extracted:

```text
CacaoAI
```

folder.

Make sure you can see:

```text
app.py
models
datasets
```

before continuing.

---

# Option 2 — Clone from GitHub

If Git is installed, cloning is recommended.

### Step 1 — Open PowerShell or Command Prompt

### Step 2 — Go to the location where you want the project

For example:

```powershell
cd Downloads
```

### Step 3 — Clone the repository

Use:

```bash
git clone https://github.com/blazingrobot/CACAO_AI.git
```

### Step 4 — Enter the project folder

```bash
cd CACAO_AI
```

You should now be inside the project folder.

Check the files with:

```bash
dir
```

You should see:

```text
app.py
models
datasets
```

---

# 🐍 Check Python Version

Before creating the environment, check Python.

Run:

```bash
python --version
```

Recommended:

```text
Python 3.12.x
```

The project was tested using:

```text
Python 3.12.10
```

If the computer has a different Python version, check the version before continuing.

---

# 🔧 Create a Virtual Environment

A virtual environment keeps the project's Python packages separate from the rest of the computer.

Inside the Cacao AI project folder, run:

```bash
python -m venv .venv
```

This creates:

```text
.venv/
```

inside the project.

---

# ▶️ Activate the Virtual Environment

## Windows PowerShell

Run:

```powershell
.venv\Scripts\Activate.ps1
```

If successful, the terminal should show something similar to:

```text
(.venv) PS C:\...\CacaoAI>
```

The `(.venv)` means the virtual environment is active.

---

## Windows Command Prompt

If using Command Prompt instead:

```cmd
.venv\Scripts\activate
```

---

# ⚠️ If PowerShell Blocks the Activation

If PowerShell displays an execution-policy error, you can use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run:

```powershell
.venv\Scripts\Activate.ps1
```

This changes the policy only for the current PowerShell session.

---

# 📦 Install Dependencies

After activating the virtual environment, install the required Python packages.

Run:

```bash
pip install streamlit pandas numpy opencv-python scikit-learn scikit-image joblib tensorflow
```

Wait for the installation to finish.

This may take some time because **TensorFlow is a large package**.

---

# 🔎 Verify the Important Packages

You can check Streamlit:

```bash
streamlit --version
```

Check TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

The development environment used for this project had TensorFlow:

```text
2.21.0
```

Scikit-learn used during development:

```text
1.6.1
```

The exact installed versions may vary slightly if packages install newer compatible versions.

---

# 🚀 Run Cacao AI

Once everything is installed and the virtual environment is activated, run:

```bash
streamlit run app.py
```

Streamlit should start the application.

The terminal will display a local address similar to:

```text
Local URL: http://localhost:8501
```

Open that address in a web browser.

You can also usually open:

```text
http://localhost:8501
```

manually.

---

# 🌐 Viewing the Website

After running:

```bash
streamlit run app.py
```

the Cacao AI website should open in the browser.

If it does not automatically open, copy the Local URL shown in the terminal and paste it into:

- Google Chrome
- Microsoft Edge
- Firefox

Example:

```text
http://localhost:8501
```

---

# 🧪 Using the Website

## Step 1 — Upload a Cacao Image

Upload a cacao image through the image upload section.

The uploaded image is used by the image-based machine-learning models.

---

## Step 2 — Analyze the Image

Click:

```text
Analyze Cacao
```

The system will process the uploaded image.

The image-based models will analyze:

1. Ripeness
2. Disease
3. Disease severity

---

# 🌿 AI Ripeness Assessment

The system uses the **Support Vector Machine (SVM)** model to classify cacao ripeness.

Possible results:

```text
Immature
Mature
Overmature
```

The system also displays the model confidence and probability distribution.

Example:

```text
Predicted Ripeness: Mature

Confidence: 76.49%
```

The SVM uses image-derived color and texture information from the cacao image.

---

# 🦠 AI Disease Detection

The system uses an **Artificial Neural Network (ANN)** to identify possible cacao diseases.

Possible classes:

```text
Healthy
Black Pod
Cacao Pod Borer
```

Example:

```text
Predicted Disease:
Cacao Pod Borer

Confidence:
98.13%
```

The model analyzes the uploaded cacao image.

---

# 📊 AI Disease Severity

The system uses **Linear Regression** to estimate disease severity.

The result is displayed as a percentage.

Example:

```text
Severity:
84.63%

Severity Level:
Severe
```

The current project classification is:

```text
1–25%   = Low
26–50%  = Medium
51–100% = Severe
```

The model derives image-based features including:

- Lesion Ratio
- Mean Brightness
- Brightness Variation

These features are calculated automatically from the uploaded image.

---

# 🔬 Additional Measurements

The system also contains a **Naive Bayes** model for fermentation quality.

Unlike the image-based models, this model currently requires additional numerical measurements:

```text
Moisture
Cadmium
Polyphenols
```

These values are not automatically obtained from the cacao image.

They represent additional laboratory or measurement information.

The user enters the values into the system.

---

# 🧪 Fermentation Quality

The **Naive Bayes** model predicts fermentation quality.

Possible outputs:

```text
Partially Fermented
Poorly Fermented
Well Fermented
```

The system also displays the confidence/probabilities of the predictions.

Example:

```text
Prediction:
Well Fermented

Confidence:
52.13%
```

---

# 🌾 Post-Harvest Information

The system also provides a **Decision Tree** model for post-harvest recommendations.

The model uses:

```text
Bean Size
Moisture Level
Moisture Percentage
```

The cacao's ripeness does NOT need to be entered manually.

Instead, the system uses the ripeness predicted by the SVM model.

This allows the image-based ripeness result to contribute to the post-harvest recommendation.

---

# 📋 Post-Harvest Recommendation

The Decision Tree provides an actionable recommendation based on the available cacao information.

Example:

```text
Controlled 4-day fermentation;
monitor fermentation closely
```

The purpose of this feature is to turn the model's analysis into a practical recommendation for post-harvest handling.

---

# 🤖 Machine Learning Models

Cacao AI currently integrates the following machine-learning algorithms:

| Algorithm | System Function | Main Input |
|---|---|---|
| **Artificial Neural Network (ANN)** | Disease Detection | Cacao Image |
| **Support Vector Machine (SVM)** | Ripeness Classification | Cacao Image |
| **Linear Regression** | Disease Severity Estimation | Cacao Image |
| **Naive Bayes** | Fermentation Quality | Moisture, Cadmium, Polyphenols |
| **Decision Tree** | Post-Harvest Recommendation | Bean Size + Moisture Information + SVM Ripeness |
| **KNN** | Planned/Not Yet Integrated | — |

---

# 🔄 System Workflow

The overall workflow of Cacao AI is:

```text
                    CACAO IMAGE
                         │
                         ▼
                ┌─────────────────┐
                │   AI ANALYSIS   │
                └─────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        SVM             ANN       Linear Regression
          │              │              │
          ▼              ▼              ▼
      Ripeness        Disease        Severity
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                Additional Information
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Naive Bayes              Decision Tree
             │                       │
             ▼                       ▼
     Fermentation Quality    Post-Harvest Recommendation
```

---

# 🎯 Purpose of the System

The main purpose of Cacao AI is to demonstrate how multiple machine-learning algorithms can be combined into one web-based cacao analysis system.

Instead of using only one model, the system assigns different machine-learning algorithms to different cacao-related tasks.

This demonstrates the practical use of machine learning in agriculture.

The system aims to help provide:

- Faster cacao assessment
- Automated image-based disease detection
- Automated ripeness classification
- Disease severity estimation
- Fermentation quality assessment
- Post-harvest guidance

The system is intended as an **AI-assisted decision-support tool**, not a replacement for professional agricultural or laboratory testing.

---

# 🌱 Why Image Analysis Is Important

The cacao image is the main input of Cacao AI.

The image allows the system to automatically analyze visual characteristics such as:

- Color
- Texture
- Visible lesions
- Visible disease symptoms
- Visual ripeness characteristics

This allows several predictions to be performed without requiring the user to manually describe the cacao.

However, not every property of cacao can currently be determined from an image.

For example:

```text
Moisture
Cadmium
Polyphenols
```

are currently provided as additional measurements because the existing Naive Bayes model was trained using these numerical features.

---

# ⚠️ Important Project Limitations

Cacao AI is a university machine-learning project and should be considered a prototype.

The predictions depend on the training data and models used by the project.

The system should not be treated as a replacement for:

- Professional agricultural diagnosis
- Laboratory testing
- Certified cacao quality assessment
- Professional disease diagnosis

In particular, the fermentation model currently relies on manually provided measurements.

---

# 🧠 Current Integration Status

| Feature | Status |
|---|---|
| Cacao Image Upload | ✅ Integrated |
| SVM Ripeness | ✅ Integrated |
| ANN Disease Detection | ✅ Integrated |
| Linear Regression Severity | ✅ Integrated |
| Naive Bayes Fermentation | ✅ Integrated |
| Decision Tree Recommendation | ✅ Integrated |
| KNN | ⏳ Not yet integrated |
| Modern Streamlit UI | 🚧 In development/finalizing |

KNN is currently excluded from the main application because its exact image feature-extraction process from training has not yet been finalized.

Do not create or claim KNN results unless it has been properly integrated and tested.

---

# 🛑 Stopping the Application

To stop the Streamlit application, go back to the terminal where it is running and press:

```text
Ctrl + C
```

The website will stop running.

To start it again later:

```bash
streamlit run app.py
```

---

# 🔁 Starting the Project Again

Every time you want to run Cacao AI:

### 1. Open the project folder

```bash
cd path/to/CacaoAI
```

### 2. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

### 4. Open the local URL

Usually:

```text
http://localhost:8501
```

---

# 📌 Quick Start

For someone who already has Python installed:

```bash
git clone https://github.com/blazingrobot/CACAO_AI.git
cd CACAO_AI
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install streamlit pandas numpy opencv-python scikit-learn scikit-image joblib tensorflow
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 👥 Project Handoff

If another group member needs to demonstrate the project:

1. Make sure the project has been downloaded/cloned completely.
2. Make sure the `models` folder is present.
3. Make sure the `datasets` folder is present.
4. Check the Python version.
5. Create and activate the virtual environment.
6. Install the required packages.
7. Run:

```bash
streamlit run app.py
```

8. Open the Streamlit local URL.
9. Upload a cacao image.
10. Click **Analyze Cacao**.
11. Review the AI results.
12. Enter the required additional measurements if demonstrating the Naive Bayes and Decision Tree features.
13. Explain the role of each machine-learning algorithm using the model table above.

---

# 🌱 Cacao AI — Project Summary

Cacao AI combines multiple machine-learning techniques into a single Streamlit-based web application for cacao analysis.

The system uses:

**ANN** for disease detection, **SVM** for ripeness classification, and **Linear Regression** for disease severity estimation using cacao images.

It also uses **Naive Bayes** to assess fermentation quality using additional measurements such as moisture, cadmium, and polyphenols.

Finally, a **Decision Tree** provides post-harvest recommendations based on cacao characteristics and the ripeness predicted by the SVM model.

The overall goal of Cacao AI is to demonstrate how machine learning can be applied to cacao agriculture by combining automated image analysis, additional measurements, and actionable recommendations into one system.

---

## 🚀 Run the Project

```bash
streamlit run app.py
```

**Cacao AI — Intelligent Cacao Analysis for Better Harvest Decisions.**
