import streamlit as st
import joblib
import numpy as np
import os

# Check if the model file exists
model_path = 'logistic_regression_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure the file is in the correct directory.")
    st.stop()

# Load the saved model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model. Please ensure the file '{model_path}' is present and valid. Error: {e}")
    st.stop()

# Load the scaler
scaler_path = 'scaler.pkl'
if not os.path.exists(scaler_path):
    st.error(f"Scaler file '{scaler_path}' not found. Please ensure the file is in the correct directory.")
    st.stop()

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Failed to load scaler. Please ensure the file '{scaler_path}' is present and valid. Error: {e}")
    st.stop()

# Define a function to make predictions
def predict_heart_disease(input_data):
    try:
        input_data_scaled = scaler.transform([input_data])
        prediction = model.predict(input_data_scaled)
        prediction_prob = model.predict_proba(input_data_scaled)[0][1]
        return prediction[0], prediction_prob
    except Exception as e:
        st.error(f"Failed to make prediction. Error: {e}")
        return None, None

# Streamlit app
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #111;
}
.stButton>button {
    color: #fff;
    background-color: #008CBA;
    border-radius: 10px;
}
div[data-baseweb="input"] {
    display: flex;
    align-items: center;
    border: 1px solid #0044cc !important; 
    box-shadow: 0 0 5px #0044cc !important; 
    border-radius: 10px;
}
div[data-baseweb="input"] input {
    border: none !important; 
    box-shadow: none !important; 
    flex-grow: 1; 
}
div[data-baseweb="input"] button {
    border: none !important; 
    box-shadow: none !important; 
}
div[data-baseweb="select"] > div {
    border: 1px solid #0044cc !important; 
    box-shadow: 0 0 5px #0044cc !important; 
    border-radius: 10px;
}
.centered-title {
    text-align: center;
    color: white;
    font-size: 2em;
}
</style>
""", unsafe_allow_html=True)


st.image('ecg_monitor.gif', caption='Muhammad Abdoola 39HXM3YQ9', use_column_width=True)

# User inputs
age = st.number_input('Age', min_value=0, max_value=100, value=50)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type (cp)', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=0, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (chol)', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ['True', 'False'])
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', ['Normal', 'Having ST-T Wave Abnormality', 'Showing probable or definite left ventricular hypertrophy'])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', ['Yes', 'No'])
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', [0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (thal)', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Convert categorical inputs to numerical values
sex = 1 if sex == 'Male' else 0
cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
fbs = 1 if fbs == 'True' else 0
restecg = ['Normal', 'Having ST-T Wave Abnormality', 'Showing probable or definite left ventricular hypertrophy'].index(restecg)
exang = 1 if exang == 'Yes' else 0
slope = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
thal = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)

# Prepare the input data for prediction
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])



# Predict heart disease based on user input
if st.button("Predict"):
    prediction, prediction_prob = predict_heart_disease(input_data)
    if prediction is not None:

        if prediction == 1:
            st.write('The patient is likely to have heart disease. 😔')
        else:
            st.write('The patient is unlikely to have heart disease. 😊')

