from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# CORS middleware ekleme

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statik dosyalar için dizin
app.mount("/static", StaticFiles(directory="static"), name="static")

# Modeli yükleme
pipeline = None  # Global değişken

def load_model():
    global pipeline
    try:
        pipeline = joblib.load('pipeline.pkl')
    except FileNotFoundError:
        pipeline = None
        print("Model file 'pipeline.pkl' not found.")

# Giriş verileri için Pydantic modeli
class InputData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

@app.get("/", response_class=HTMLResponse)
async def read_index():
    load_model()  # Modeli yükle
    with open('static/index.html', 'r') as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/bmi", response_class=HTMLResponse)
async def read_bmi_page():
    with open('static/bmi.html', 'r') as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/predict/")
async def predict(data: InputData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
        
    # Verileri DataFrame'e dönüştürme
    data_dict = data.dict()
    data_dicts = {
        'Pregnancies': data_dict['pregnancies'],
        'Glucose': data_dict['glucose'],
        'BloodPressure': data_dict['blood_pressure'],
        'SkinThickness': data_dict['skin_thickness'],
        'Insulin': data_dict['insulin'],
        'BMI': data_dict['bmi'],
        'DiabetesPedigreeFunction': data_dict['diabetes_pedigree_function'],
        'Age': data_dict['age'],
        'New_BMI_Category': categorize_bmi(data_dict['bmi']),
        'New_Glucose_Class': classify_glucose(data_dict['glucose'])
    }

    X_user = pd.DataFrame([data_dicts])
    
    try:
        # Tahmin yapma
        prediction = pipeline.predict(X_user)
        result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"prediction": result}

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

