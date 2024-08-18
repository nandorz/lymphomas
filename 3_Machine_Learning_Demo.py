import streamlit as st
import pandas as pd
import pickle

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('data LNH final.csv', delimiter=';')

# Initialize data and model
data = load_data()

# Load the trained model
model = pickle.load(open('model/logistic_regression_model.pkl', 'rb'))

# Streamlit app title
st.title('Demo: Machine Learning')

st.header(""" 
    Fill in the patient information below and we'll predict the 2-year mortality outcome!
""")

# Input fields
age = st.number_input('Age (year):', min_value=0, max_value=150, value=60, step=1)
sex = st.radio('Gender:', ['Male', 'Female'])
sex = 1 if sex == 'Male' else 0
hb = st.number_input('Haemoglobin (g/dL):', min_value=0.0, value=15.0, format="%.2f")
leu = st.number_input('Leukocyte Count (/mm^3):', min_value=0.0, value=8000.0, format="%.2f") / 1000
neu = st.number_input('Absolute Neutrophil Count (/mm^3):', min_value=0.0, value=5000.0, format="%.2f") / 1000
lim = st.number_input('Absolute Lymphocyte Count (/mm^3):', min_value=0.0, value=1500.0, format="%.2f") / 1000
mon = st.number_input('Absolute Monocyte Count (/mm^3):', min_value=0.0, value=800.0, format="%.2f") / 1000

if leu < neu + lim + mon:
    st.markdown(
        '<p style="color:red;">Warning!! Please recheck your blood count input</p>',
        unsafe_allow_html=True
    )

plt = st.number_input('Absolute Platelet Count (/mm^3):', min_value=0.0, value=300000.0, format="%.2f") / 1000
ldh = st.number_input('LDH level in blood (unit/L):', min_value=0.0, value=150.0, format="%.2f")
ldh = 0 if ldh < 200 else 1  # Binary conversion

b_symptoms = st.radio('B-Symptoms:', ['Not Present', 'Present'])
b_symptoms = 0 if b_symptoms == 'Not Present' else 1

extranodal = st.radio('Extranodal Involvement:', ['0 or 1', 'More Than 1'])
extranodal = 0 if extranodal == '0 or 1' else 1

ecog = st.radio('ECOG Performance Status:', ['Good', 'Bad'])
ecog = 0 if ecog == 'Bad' else 1

stage = st.radio('Ann-Arbor Stage:', ['I', 'II', 'III', 'IV'])
stage = 0 if stage in ['I', 'II'] else 1

response = st.radio('R-CHOP Treatment Response:', ['Not Response', 'Response'])
response = 0 if response == 'Not Response' else 1

# Calculate 'ipi' as needed
ipi = sex + stage + response + ecog + ldh + extranodal

patient_data = pd.DataFrame({
    'sex': [sex],
    'stage': [stage],
    'response': [response],
    'b_symptoms': [b_symptoms],
    'ecog': [ecog],
    'extranodal': [extranodal],
    'ldh': [ldh],
    'ipi': [ipi],
    'age': [age],
    'hb': [hb],
    'leu': [leu],
    'neu': [neu],
    'lim': [lim],
    'mon': [mon],
    'plt': [plt],
})

if st.button('Evaluate 2-Year Mortality'):
    prognosis = model.predict(patient_data)
    prognosis_text = "Survive" if prognosis[0] == 0 else "Not Survive"
    color = "green" if prognosis[0] == 0 else "red"
    st.markdown(
        f'<h3 style="color:{color};">Based on the data input, the 2-year mortality prediction is: {prognosis_text}</h3>',
        unsafe_allow_html=True
    )    
