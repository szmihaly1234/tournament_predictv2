import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load all artifacts
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model('nn_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('onehot_columns.pkl', 'rb') as f:
        onehot_columns = pickle.load(f)
    
    # Load team lists from original data
    df = pd.read_csv('all_matches.csv')
    unique_teams = sorted(list(set(df['home_team'].unique()).union(set(df['away_team'].unique()))))
    
    return model, le, scaler, onehot_columns, unique_teams

# Betöltjük az artefaktokat
model, le, scaler, onehot_columns, unique_teams = load_artifacts()

# Create the Streamlit UI
st.title("Football Tournament Predictor")
st.write("Predict which tournament a football match belongs to based on match details")

# Input form
with st.form("match_details"):
    home_team = st.selectbox("Home Team", unique_teams)
    away_team = st.selectbox("Away Team", unique_teams)
    
    match_date = st.date_input("Match Date")  # Normál dátumbeviteli mező
    year = match_date.year
    month = match_date.month
    day = match_date.day  # Az újonnan bevezetett nap érték

    result = st.selectbox("Expected Result", ["home_win", "away_win", "draw"], index=0)
    model_choice = st.selectbox("Choose Model", ["Random Forest", "Neural Network"])
    
    submitted = st.form_submit_button("Predict Tournament")

if submitted:
    try:
        # Készítsünk egy DataFrame-et a megfelelő oszlopokkal
        X = pd.DataFrame(0, index=[0], columns=onehot_columns)

        # Beállítjuk a megfelelő one-hot kódolt oszlopokat
        home_col = f"home_team_{home_team}"
        away_col = f"away_team_{away_team}"
        result_col = f"result_{result}"

        if home_col in X.columns:
            X[home_col] = 1
        else:
            st.warning(f"Home team '{home_team}' not in training data. Using default values.")
        
        if away_col in X.columns:
            X[away_col] = 1
        else:
            st.warning(f"Away team '{away_team}' not in training data. Using default values.")
        
        if result_col in X.columns:
            X[result_col] = 1

        # Beállítjuk az egyéb numerikus jellemzőket
        X['year'] = year
        X['month'] = month
        X['day'] = day  # Az új nap beállítása
        
        # Csak az eredeti tréning során használt oszlopokat tartjuk meg
        X = X[onehot_columns]

        # Standardizálás
        X_scaled = scaler.transform(X)

        # Biztosítsuk, hogy a neurális hálózat input mérete megfelelő legyen
        if model_choice == "Neural Network":
            X_scaled = np.expand_dims(X_scaled, axis=0)

        # Model előrejelzés
        if model_choice == "Random Forest":
            preds = rf_model.predict_proba(X_scaled)[0]
        else:
            preds = model.predict(X_scaled)[0]

        # Top 3 legvalószínűbb eredmény visszaalakítása
        top3_idx = np.argsort(preds)[-3:][::-1]
        valid_indices = [i for i in top3_idx if i in le.classes_]

        if valid_indices:
            top3_tournaments = le.inverse_transform(valid_indices)
        else:
            top3_tournaments = ["Unknown" for _ in range(len(top3_idx))]

        top3_probs = preds[top3_idx]

        # Eredmények kiírása
        st.subheader("Prediction Results")
        st.write(f"Most likely tournament: **{top3_tournaments[0]}** ({(top3_probs[0]*100):.1f}%)")

        st.write("Top 3 predicted tournaments:")
        for tourn, prob in zip(top3_tournaments, top3_probs):
            st.write(f"- {tourn}: {(prob*100):.1f}%")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please try different input values.")

# Oldalsáv információ
st.sidebar.markdown("""
**About this app:**
This app predicts which football tournament a match belongs to based on:
- Home and away teams
- Match date (year, month, day)
- Match result
- Machine learning models (Random Forest or Neural Network)

The model was trained on historical international football match data.
""")
