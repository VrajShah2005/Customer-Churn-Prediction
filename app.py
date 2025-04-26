import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------ STYLING ------------------
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: white;
        }

        h1, h2, h3, h4, h5 {
            color: #FFD700 !important;
            animation: title-glow 1.5s ease-in-out infinite alternate;
            text-shadow: 0 0 10px #FFD700, 0 0 20px #FFD700;
        }

        label, .stSelectbox label, .stSlider label, .stNumberInput label {
            font-size: 30px;
            color: #FFFFFF !important;
            font-weight: bold;
            animation: text-fade 2s ease-in-out infinite alternate;
            text-shadow: 0 0 5px #ffffff88;
        }

        .stButton > button {
            background-color: #FFD700;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            box-shadow: 0 0 15px #FFD700;
            transition: 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #FFC300;
            transform: scale(1.05);
            box-shadow: 0 0 25px #FFD700, 0 0 30px #FFD700;
        }

        .safe-alert {
            color: #FFD700;
            font-size: 20px;
            font-weight: bold;
            animation: glow 1.5s ease-in-out infinite alternate;
            text-shadow: 0 0 10px #FFD700, 0 0 20px #FFD700;
        }

        .churn-alert {
            color: #FF4C4C;
            font-size: 20px;
            font-weight: bold;
            animation: pulse 1.2s infinite;
            text-shadow: 0 0 10px #FF4C4C, 0 0 20px #FF4C4C;
        }

        @keyframes glow {
            from { text-shadow: 0 0 5px #FFD700, 0 0 10px #FFD700; }
            to { text-shadow: 0 0 20px #FFD700, 0 0 30px #FFD700; }
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.03); }
            100% { transform: scale(1); }
        }

        @keyframes title-glow {
            from { text-shadow: 0 0 10px #FFD700; }
            to { text-shadow: 0 0 20px #FFD700, 0 0 30px #FFD700; }
        }

        @keyframes text-fade {
            from { opacity: 0.8; }
            to { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title('✨ Customer Churn Prediction')
st.markdown('<p style="font-size:20px; font-weight:bold; color:white;">Made By : Vraj Shah</p>', unsafe_allow_html=True)
st.subheader("📋 Customer Information")

# ------------------ INPUT FIELDS ------------------
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure (Years)', 0, 10)
num_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# ------------------ DATA PREP ------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encoding
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure correct columns
expected_features = scaler.feature_names_in_
missing_cols = set(expected_features) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[expected_features]

# Scale data
input_data_scaled = scaler.transform(input_data)

# ------------------ PREDICTION ------------------
if st.button('🚀 Predict Churn'):
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        st.markdown(f"""
            <p class="churn-alert">
                ⚠️ The customer is likely to churn. (Confidence: {prediction_proba:.2f})
            </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <p class="safe-alert">
                ✅ The customer is not likely to churn. (Confidence: {1 - prediction_proba:.2f})
            </p>
        """, unsafe_allow_html=True)
