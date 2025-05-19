import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# Set page config
st.set_page_config(page_title="Tournament Predictor", layout="wide")

# Load the saved model and preprocessing objects
@st.cache_resource
def load_assets():
    model = load_model('tournament_model.h5')
    scaler = joblib.load('scaler.save')
    label_encoder = joblib.load('label_encoder.save')
    return model, scaler, label_encoder

model, scaler, label_encoder = load_assets()

# Create inverse mapping for tournament labels
tournament_mapping = {i: name for i, name in enumerate(label_encoder.classes_)}

# App title and description
st.title("Football Tournament Predictor")
st.markdown("""
Predict the type of tournament based on match characteristics.
""")

# Sidebar for input features
st.sidebar.header("Input Match Parameters")

# Create input widgets
with st.sidebar.form("input_form"):
    result = st.selectbox("Match Result", ["home_win", "away_win", "draw"])
    year = st.number_input("Year", min_value=1970, max_value=datetime.now().year, value=2023)
    month = st.selectbox("Month", range(1, 13), format_func=lambda x: datetime(1900, x, 1).strftime('%B'))
    
    submitted = st.form_submit_button("Predict Tournament")

# Main content area
if submitted:
    # Prepare input data
    input_data = pd.DataFrame({
        'results': [result],
        'year': [year],
        'month': [month]
    })
    
    # Encode and scale the input
    input_data['results'] = label_encoder.transform(input_data['results'])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_tournament = tournament_mapping[predicted_class[0]]
    confidence = np.max(prediction) * 100
    
    # Display results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Tournament", predicted_tournament)
        st.metric("Confidence", f"{confidence:.1f}%")
    
    with col2:
        st.write("**Probability Distribution:**")
        prob_df = pd.DataFrame({
            'Tournament': [tournament_mapping[i] for i in range(len(prediction[0]))],
            'Probability': prediction[0]
        }).sort_values('Probability', ascending=False)
        
        st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}), hide_index=True)
    
    # Visualize probabilities
    st.bar_chart(prob_df.set_index('Tournament'))

# Add some explanations
st.markdown("""
### How It Works
1. Select the match result, year, and month
2. Click the "Predict Tournament" button
3. View the predicted tournament type and confidence level

The model was trained on historical match data to predict the type of tournament based on these simple features.
""")
