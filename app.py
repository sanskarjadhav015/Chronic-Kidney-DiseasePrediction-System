import streamlit as st
import pandas as pd
import pickle

# ================= LOAD MODEL & SCALER =================
scaler = pickle.load(open("models/scaler.pkl", "rb"))
model_gbc = pickle.load(open("models/model_gbc.pkl", "rb"))

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
    df['cad'] = df['cad'] = df['cad'].map({'yes': 1, 'no': 0})
    df['appet'] = df['appet'].map({'good': 1, 'poor': 0})
    df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})

    # Scale numeric values
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Predict
    prediction = model_gbc.predict(df)

    return prediction[0]


# ================= STREAMLIT UI =================
st.title("Chronic Kidney Disease Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=48)
    bp = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)
    sg = st.number_input("Specific Gravity", min_value=1.005, max_value=1.050, value=1.020)
    al = st.number_input("Albumin", min_value=0.0, max_value=5.0, value=1.0)
    hemo = st.number_input("Hemoglobin", min_value=5.0, max_value=20.0, value=15.4)
    sc = st.number_input("Serum Creatinine", min_value=0.5, max_value=10.0, value=1.2)

with col2:
    htn = st.selectbox("Hypertension", ["yes", "no"])
    dm = st.selectbox("Diabetes", ["yes", "no"])
    cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
    appet = st.selectbox("Appetite", ["good", "poor"])
    pc = st.selectbox("Protein in Urine", ["normal", "abnormal"])

# ================= PREDICT BUTTON =================
if st.button("Predict"):
    result = predict_chronic_disease(
        age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc
    )

    if result == 1:
        st.success("✅ The patient has Chronic Kidney Disease (CKD).")
    else:
        st.success("✅ The patient does NOT have Chronic Kidney Disease (CKD).")

## .venv\Scripts\activate
##  streamlit run app.py
