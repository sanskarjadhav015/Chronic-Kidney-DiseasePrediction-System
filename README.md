# 🏥 Chronic Kidney Disease (CKD) AI Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application designed to evaluate patient risk of **Chronic Kidney Disease (CKD)** using 11 key clinical and laboratory indicators.

> 📖 **Need a complete walkthrough to explain this project to others?** Check out the [Comprehensive Project Explanation Guide](file:///d:/portfolio_project/Chronic-Kidney-DiseasePrediction-System/PROJECT_EXPLANATION.md) for 30-second elevator pitches, feature definitions, ML pipeline step-by-step breakdowns, and top interview Q&As!

---

## 🌟 Key Features

- **Leak-Free ML Pipeline**: Data preprocessing pipeline trained using `MinMaxScaler` and `SMOTE` oversampling strictly on training splits to ensure zero data leakage.
- **High Diagnostic Accuracy**: Powered by a **Gradient Boosting Classifier** reaching **98.75% accuracy** on unseen test evaluations.
- **Streamlit Web Dashboard**: Responsive UI with dynamic input fields, metric tooltips, cached model inference (`@st.cache_resource`), and visual risk banners.
- **Automated Testing Suite**: Includes `unittest` test suite verifying pipeline transformations and binary classification outputs.

---

## 🔬 Clinical Input Features

The model evaluates 11 clinically significant attributes:

| Feature | Description | Type / Scale | Normal Clinical Range |
| :--- | :--- | :--- | :--- |
| `age` | Patient Age | Numerical (Years) | 1 – 120 |
| `bp` | Blood Pressure | Numerical (mm/Hg) | 60 – 120 mm/Hg |
| `sg` | Specific Gravity | Categorical | 1.005 – 1.025 |
| `al` | Albumin Level | Categorical (0 - 5) | 0.0 (Absence of excess protein) |
| `hemo` | Hemoglobin Level | Numerical (g/dL) | 12.0 – 18.0 g/dL |
| `sc` | Serum Creatinine | Numerical (mg/dL) | 0.6 – 1.2 mg/dL |
| `htn` | Hypertension | Binary (`yes`/`no`) | `no` |
| `dm` | Diabetes Mellitus | Binary (`yes`/`no`) | `no` |
| `cad` | Coronary Artery Disease | Binary (`yes`/`no`) | `no` |
| `appet` | Patient Appetite | Binary (`good`/`poor`) | `good` |
| `pc` | Pus Cells in Urine | Binary (`normal`/`abnormal`) | `normal` |

---

## 📊 Model Evaluation & Benchmarking

Eight machine learning algorithms were benchmarked on the test dataset:

| Model Algorithm | Accuracy | Precision (CKD / Not CKD) | Recall (CKD / Not CKD) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier (Deployed)** | **98.75%** | **1.00 / 0.97** | **0.98 / 1.00** | **0.99** |
| Decision Tree Classifier | 98.75% | 1.00 / 0.97 | 0.98 / 1.00 | 0.99 |
| Random Forest Classifier | 97.50% | 1.00 / 0.94 | 0.96 / 1.00 | 0.98 |
| Support Vector Classifier (SVC) | 96.25% | 1.00 / 0.91 | 0.94 / 1.00 | 0.97 |
| K-Nearest Neighbors (KNN) | 96.25% | 1.00 / 0.91 | 0.94 / 1.00 | 0.97 |
| AdaBoost Classifier | 96.25% | 1.00 / 0.91 | 0.94 / 1.00 | 0.97 |
| Logistic Regression | 95.00% | 1.00 / 0.88 | 0.92 / 1.00 | 0.96 |
| Gaussian Naive Bayes | 95.00% | 1.00 / 0.88 | 0.92 / 1.00 | 0.96 |

---

## 🛠️ Project Architecture & Structure

```
Chronic-Kidney-DiseasePrediction-System/
├── models/                     # Serialized Model Artifacts
│   ├── model_gbc.pkl           # Trained Gradient Boosting Classifier
│   └── scaler.pkl              # Fitted MinMaxScaler Instance
├── app.py                      # Streamlit Web Application & Interface
├── train_model.py              # Automated, Leak-Free Retraining Script
├── test_pipeline.py            # Unit Tests Suite for Inference & Pipeline
├── CKD_Prediction.ipynb        # EDA & Model Experimentation Notebook
├── kidney_disease.csv          # Dataset (UCI Chronic Kidney Disease)
├── requirements.txt            # Pinned Project Dependencies
└── .gitignore                  # Git Exclusion Rules
```

---

## ⚡ Quickstart & Installation

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/sanskarjadhav015/Chronic-Kidney-DiseasePrediction-System.git
cd Chronic-Kidney-DiseasePrediction-System

# Install required Python packages
pip install -r requirements.txt
```

### 2. Run Model Retraining (Optional)
To retrain the model and update serialized pickles:
```bash
python train_model.py
```

### 3. Run Automated Tests
```bash
python -m unittest test_pipeline.py
```

### 4. Launch Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📄 License
This project is open-source under the MIT License.
