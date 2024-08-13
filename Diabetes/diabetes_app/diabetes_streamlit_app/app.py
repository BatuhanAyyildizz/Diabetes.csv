import streamlit as st
import joblib
import pandas as pd

# Modeli yükleme
pipeline = None  # Global değişken

def load_model():
    global pipeline
    try:
        pipeline = joblib.load('pipeline.pkl')
    except FileNotFoundError:
        pipeline = None
        st.error("Model file 'pipeline.pkl' not found.")

# Glukoz ve BMI kategorileri
def classify_glucose(glucose):
    if glucose < 100:
        return 0  # 'Normal'
    elif glucose < 125:
        return 1  # 'Prediabetes'
    else:
        return 2  # 'Diabetes'

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 0  # 'Underweight'
    elif bmi < 24.9:
        return 1  # 'Healthy'
    elif bmi < 29.9:
        return 2  # 'Overweight'
    else:
        return 3  # 'Obese'

# Streamlit uygulaması
st.title("Diabetes Prediction App")

# Modeli yükle
load_model()

if pipeline is not None:
    # Kullanıcıdan giriş verilerini al
    pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose", min_value=0.0, value=0.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=0.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=0.0)
    insulin = st.number_input("Insulin", min_value=0.0, value=0.0)
    bmi = st.number_input("BMI", min_value=0.0, value=0.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.0)
    age = st.number_input("Age", min_value=0, value=0)

    if st.button("Predict"):
        # Verileri DataFrame'e dönüştürme
        data_dicts = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree_function,
            'Age': age,
            'New_BMI_Category': categorize_bmi(bmi),
            'New_Glucose_Class': classify_glucose(glucose)
        }

        X_user = pd.DataFrame([data_dicts])
        
        try:
            # Tahmin yapma
            prediction = pipeline.predict(X_user)
            result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.error("Model not loaded.")
