
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('best_salary_model.pkl')

# Title
st.title('Salary Prediction App')
st.write('Enter your details to predict your salary.')

# Inputs based on our features
age = st.number_input('Age', min_value=18, max_value=100, value=30)
gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x==1 else 'Female')
education = st.selectbox('Education Level', options=[0, 1, 2], format_func=lambda x: ["Bachelor's", "Master's", "PhD"][x])
job_title = st.number_input('Job Title (Encoded ID)', min_value=0, max_value=173, value=159)
experience = st.number_input('Years of Experience', min_value=0, max_value=50, value=5)

if st.button('Predict Salary'):
    features = np.array([[age, gender, education, job_title, experience]])
    prediction = model.predict(features)
    st.success(f'The predicted salary is ${prediction[0]:,.2f}')
