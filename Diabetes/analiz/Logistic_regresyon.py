'''Burda amaçladığım diabetes.ipynb dosyasında dataseti ve modellerin bu dataset üzerindeki işlemlerini ve sonuçlarını inceledim ve logistic regresyonu seçerek bir uygulama geliştirmeye karar verdim bunun için bu modeli scale ettim ve kaydettim ilerleyen süreçte bu pkl uzantılı dosyayı kullanarak uygulamamı geliştireceğim'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.pipeline import Pipeline
import joblib


#Glucose için yeni bir özellik
def classify_glucose(glucose):
    if glucose < 100:
        return 0 #'Normal'
    elif glucose < 125:
        return 1 #'Prediabetes'
    else:
        return 2 #'Diabetes'



# BMI için yeni kategoriler
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 0 #'Underweight'
    elif bmi < 24.9:
        return  1#'Healthy'
    elif bmi < 29.9:
        return 2 #'Overweight'
    else:
        return 3#'Obese'

df = pd.read_csv('diabetes.csv')
df['New_BMI_Category'] = df['BMI'].apply(categorize_bmi)
df['New_Glucose_Class'] = df['Glucose'].apply(classify_glucose)

df_copy = df.copy(deep=True)

# 0 değerlerini NaN ile değiştirme
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

data_cleaned = df_copy.dropna()
df_copy=data_cleaned

X = data_cleaned.drop(columns='Outcome', axis=1)
y = data_cleaned['Outcome']

# Özellikleri ölçekleme ve modeli tanımlama
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('logistic', LogisticRegression(class_weight='balanced'))
])

# Eğitim ve test verilerine ayırma
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, stratify=y, random_state=2)

# Pipeline'ı fit etme
pipeline.fit(train_x, train_y)

# Performans değerlendirme
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_pred = pipeline.predict(train_x)
test_pred = pipeline.predict(test_x)

train_accuracy = accuracy_score(train_y, train_pred)
test_accuracy = accuracy_score(test_y, test_pred)
print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
print(f"\nClassification Report (Test Set):\n", classification_report(test_y, test_pred))

# Confusion Matrix
cm = confusion_matrix(test_y, test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Pipeline'ı kaydetme
joblib.dump(pipeline, 'pipeline.pkl')


