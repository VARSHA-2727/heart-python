import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_chat import message
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define a function to train and save the model and scaler
def train_and_save_model():
    df = pd.read_csv('heart_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    return model, scaler

# Check if the model and scaler files exist, if not, train and save them
if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
    st.warning("Model and scaler not found. Training and saving a new model...")
    model, scaler = train_and_save_model()
else:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')

# Title of the application
st.title("Heart Condition Chatbot")

# Initialize session state for messages and conversation context
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['conversation_stage'] = 'initial'
    st.session_state['user_data'] = {}
    st.session_state['user_inputs'] = []

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

# Function to predict heart condition
def predict_heart_condition(user_data):
    user_data_array = np.array([
        user_data['age'], user_data['sex'], user_data['cp'], user_data['trestbps'],
        user_data['chol'], user_data['fbs'], user_data['restecg'], user_data['thalach'],
        user_data['exang'], user_data['oldpeak'], user_data['slope'], user_data['ca'], user_data['thal']
    ]).reshape(1, -1)

    user_data_scaled = scaler.transform(user_data_array)
    prediction = model.predict(user_data_scaled)[0]

    # Debugging: Print user data and model prediction
    print("User Data:", user_data)
    print("Model Prediction (raw):", prediction)

    if prediction == 1:
        return "You are at risk of heart disease. Please consult a doctor."
    else:
        return "You are likely to be healthy. However, regular checkups are recommended."

# Function to handle user messages
def handle_message():
    user_input = st.session_state.user_input

    if user_input:
        st.session_state.user_inputs.append(user_input)
        st.session_state.user_input = ""

        # Input handling with validation
        try:
            if 'age' not in st.session_state['user_data']:
                st.session_state['user_data']['age'] = float(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Great! Now, please enter your sex (Male/Female)."})
            elif 'sex' not in st.session_state['user_data']:
                if user_input.lower() in ["male", "female"]:
                    st.session_state['user_data']['sex'] = 1 if user_input.lower() == "male" else 0
                    st.session_state['messages'].append({"role": "user", "content": user_input})
                    st.session_state['messages'].append({"role": "assistant", "content": "Thank you. Now, please enter your chest pain type (0-3)."})
                else:
                    st.session_state['messages'].append({"role": "assistant", "content": "Please enter Male or Female for your sex."})
            elif 'cp' not in st.session_state['user_data']:
                st.session_state['user_data']['cp'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Please enter your resting blood pressure."})
            elif 'trestbps' not in st.session_state['user_data']:
                st.session_state['user_data']['trestbps'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Please enter your cholesterol level."})
            elif 'chol' not in st.session_state['user_data']:
                st.session_state['user_data']['chol'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Is your fasting blood sugar > 120 mg/dl? (0 for No, 1 for Yes)."})
            elif 'fbs' not in st.session_state['user_data']:
                st.session_state['user_data']['fbs'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Enter your resting ECG result (0-2)."})
            elif 'restecg' not in st.session_state['user_data']:
                st.session_state['user_data']['restecg'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Enter your maximum heart rate achieved."})
            elif 'thalach' not in st.session_state['user_data']:
                st.session_state['user_data']['thalach'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Do you have exercise induced angina? (0 for No, 1 for Yes)."})
            elif 'exang' not in st.session_state['user_data']:
                st.session_state['user_data']['exang'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Enter the ST depression induced by exercise relative to rest."})
            elif 'oldpeak' not in st.session_state['user_data']:
                st.session_state['user_data']['oldpeak'] = float(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Enter the slope of the peak exercise ST segment (0-2)."})
            elif 'slope' not in st.session_state['user_data']:
                st.session_state['user_data']['slope'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Enter the number of major vessels colored by fluoroscopy (0-3)."})
            elif 'ca' not in st.session_state['user_data']:
                st.session_state['user_data']['ca'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['messages'].append({"role": "assistant", "content": "Enter the thalassemia status (0-3)."})
            elif 'thal' not in st.session_state['user_data']:
                st.session_state['user_data']['thal'] = int(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                
                # Call prediction after collecting all data
                result = predict_heart_condition(st.session_state['user_data'])
                st.session_state['messages'].append({"role": "assistant", "content": result})
                st.session_state['conversation_stage'] = 'finished'
        except ValueError:
            st.session_state['messages'].append({"role": "assistant", "content": "Please enter valid numeric values where required."})

# Display chat messages
for i, msg in enumerate(st.session_state['messages']):
    message(msg["content"], is_user=(msg["role"] == "user"), key=f"msg_{i}")

# Start the conversation
if st.session_state['conversation_stage'] == 'initial':
    st.session_state['messages'].append({"role": "assistant", "content": "Hello! I can help you assess your risk of heart disease. Let's start with your age."})
    st.session_state['conversation_stage'] = 'collecting_data'

# Display the input box for user messages
if st.session_state['conversation_stage'] == 'collecting_data':
    st.text_input("You:", key="user_input", on_change=handle_message)

if st.session_state['conversation_stage'] == 'finished':
    st.session_state['messages'].append({"role": "assistant", "content": "If you need further assistance, please restart the conversation."})
