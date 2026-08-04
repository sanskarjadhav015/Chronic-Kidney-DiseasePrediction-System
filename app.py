import os
import pickle
import pandas as pd
import streamlit as st

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="Chronic Kidney Disease AI Diagnostic System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= LOAD MODEL & SCALER WITH CACHING =================
@st.cache_resource
def load_ml_assets():
    scaler_path = os.path.join("models", "scaler.pkl")
    model_path = os.path.join("models", "model_gbc.pkl")
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        st.error("❌ Model files not found! Please run 'python train_model.py' first.")
        st.stop()
        
    scaler = pickle.load(open(scaler_path, "rb"))
    model = pickle.load(open(model_path, "rb"))
    return scaler, model

scaler, model_gbc = load_ml_assets()

# ================= PREDICTION FUNCTION =================
def predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc):
    df_dict = {
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'hemo': [hemo],
        'sc': [sc],
        'htn': [htn],
        'dm': [dm],
        'cad': [cad],
        'appet': [appet],
        'pc': [pc]
    }

    df = pd.DataFrame(df_dict)

    # Encode categorical values
    df['htn'] = df['htn'].map({'yes': 1, 'no': 0})
    df['dm'] = df['dm'].map({'yes': 1, 'no': 0})
    df['cad'] = df['cad'].map({'yes': 1, 'no': 0})
    df['appet'] = df['appet'].map({'good': 1, 'poor': 0})
    df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})

    # Scale numeric values
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Predict class and probability if available
    prediction = model_gbc.predict(df)[0]
    probabilities = model_gbc.predict_proba(df)[0] if hasattr(model_gbc, "predict_proba") else None
    
    return prediction, probabilities

# ================= SIDEBAR METADATA & INFO =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/kidney.png", width=70)
    st.title("System Info")
    st.markdown("""
    **Model**: Gradient Boosting Classifier
    **Accuracy**: ~98.75%
    **Optimization**: SMOTE + MinMax Scaling
    **Inputs**: 11 Clinical Features
    """)
    st.divider()
    st.markdown("### ⚠️ Clinical Notice")
    st.info(
        "This system provides AI-assisted risk analysis based on clinical indicators. "
        "It is designed for preliminary screening and should not replace professional medical diagnosis."
    )

# ================= MAIN APP HEADER =================
st.title("🏥 Chronic Kidney Disease (CKD) AI Assessor")
st.markdown(
    "Enter patient clinical vitals, laboratory results, and medical history below to obtain an instant AI-powered risk evaluation."
)
st.divider()

# ================= INPUT FORM LAYOUT =================
with st.form("ckd_prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🩸 Vitals & Laboratory Metrics")
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=48, help="Patient age in years")
        bp = st.number_input("Blood Pressure (mm/Hg)", min_value=40, max_value=200, value=80, help="Resting blood pressure")
        sg = st.selectbox("Specific Gravity (sg)", options=[1.005, 1.010, 1.015, 1.020, 1.025], index=3, help="Urine specific gravity level")
        al = st.selectbox("Albumin Level (al)", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=1, help="Urine albumin level (0-5 scale)")
        hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=22.0, value=15.4, step=0.1, help="Hemoglobin level in blood (Normal: 12-18 g/dL)")
        sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.2, step=0.1, help="Serum creatinine level (Normal: 0.6-1.2 mg/dL)")

    with col2:
        st.subheader("📋 Clinical History & Physical Indicators")
        htn = st.selectbox("Hypertension (HTN)", options=["no", "yes"], index=1, help="History of high blood pressure")
        dm = st.selectbox("Diabetes Mellitus (DM)", options=["no", "yes"], index=1, help="History of diabetes")
        cad = st.selectbox("Coronary Artery Disease (CAD)", options=["no", "yes"], index=0, help="History of coronary artery disease")
        appet = st.selectbox("Appetite Status", options=["good", "poor"], index=0, help="Patient overall appetite")
        pc = st.selectbox("Pus Cells in Urine (PC)", options=["normal", "abnormal"], index=0, help="Pus cell presence in urinalysis")
        
    st.divider()
    submit_button = st.form_submit_button("🔍 Run CKD Risk Assessment", use_container_width=True)

# ================= PREDICTION RESULTS =================
if submit_button:
    try:
        result, proba = predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc)
        
        st.subheader("📊 Diagnostic Summary & Assessment")
        
        if result == 1:
            ckd_prob = proba[1] * 100 if proba is not None else None
            st.error("🚨 **HIGH RISK DETECTED: Patient is classified with Chronic Kidney Disease (CKD)**")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Predicted Condition", "CKD Positive (Risk Detected)", delta="- High Alert", delta_color="inverse")
            with res_col2:
                if ckd_prob is not None:
                    st.metric("AI Confidence Score", f"{ckd_prob:.1f}%")
            
            st.warning("👉 **Recommended Action**: Consult a certified Nephrologist for detailed diagnostic evaluation (e.g. eGFR test, renal ultrasound).")
        else:
            not_ckd_prob = proba[0] * 100 if proba is not None else None
            st.success("✅ **LOW RISK DETECTED: Patient shows NO signs of Chronic Kidney Disease (Not CKD)**")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Predicted Condition", "No CKD Detected", delta="+ Normal Risk", delta_color="normal")
            with res_col2:
                if not_ckd_prob is not None:
                    st.metric("AI Confidence Score", f"{not_ckd_prob:.1f}%")
                    
            st.info("👉 **Recommended Action**: Maintain a healthy lifestyle with regular blood pressure and blood glucose checkups.")
            
    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {str(e)}")
