import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

# Modellek és skálázó betöltése
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
nn_model = tf.keras.models.load_model("nn_model.h5")
print("Neurális hálózat bemeneti alakja:", nn_model.input_shape)

# A csapatok listájának beállítása (példaértékek, frissítsd a dataset alapján)
teams = ["Brazil", "Argentina", "France", "Germany", "England", "Spain", "Italy", "Portugal"]

st.title("Match Outcome Prediction")

st.markdown("""
Ez az alkalmazás lehetővé teszi, hogy előre jelezd a mérkőzés eredményét a Random Forest vagy a Neurális Hálózat modellekkel.
Válaszd ki az alábbi adatokat a predikcióhoz!
""")

# Csapatok kiválasztása
home_team = st.selectbox("Hazai csapat", teams)
away_team = st.selectbox("Vendég csapat", teams)

# Dátum kiválasztása
match_date = st.date_input("Mérkőzés dátuma")

# Eredmény kiválasztása
results_options = {"Döntetlen": 0, "Hazai győzelem": 1, "Vendég győzelem": 2}
results_choice = st.selectbox("Eredmény", list(results_options.keys()))

# Modell kiválasztása
model_choice = st.selectbox("Válassz egy modellt", ["Random Forest", "Neurális Hálózat"])

# Adatok előkészítése
year = match_date.year
month = match_date.month
results_val = results_options[results_choice]

# Adatok skálázása
input_data = np.array([[results_val, year, month]])
input_data_scaled = scaler.transform(input_data)
print("Input shape:", input_data_scaled.shape)

# LabelEncoder betöltése, ha korábban használtuk
import pickle
le = pickle.load(open("label_encoder.pkl", "rb"))




if st.button("Előrejelzés"):
    if model_choice == "Random Forest":
        probs = rf_model.predict_proba(input_data_scaled)[0]
    else:
        probs = nn_model.predict(input_data_scaled)[0]
    
    # Top 3 legvalószínűbb eredmény
    top_3_indices = np.argsort(probs)[-3:][::-1]
    top_3_labels = [f"Tournament {i}" for i in top_3_indices]
    top_3_probs = [probs[i] for i in top_3_indices]
    
# Top 3 előrejelzés visszaalakítása
top_3_tournaments = le.inverse_transform(top_3_indices)
st.markdown("### Előrejelzések:")
for i in range(3):
    st.write(f"**{top_3_tournaments[i]}:** {top_3_probs[i]:.2f} valószínűséggel")
