import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

# Modellek és skálázó betöltése
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
nn_model = tf.keras.models.load_model("nn_model.h5")
le = pickle.load(open("label_encoder.pkl", "rb"))  # LabelEncoder betöltése

# A csapatok listájának beállítása (frissítsd a dataset alapján)
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

# Feature-ek előkészítése az eredeti hálózat bemeneti alakjához
year = match_date.year
month = match_date.month
results_val = results_options[results_choice]

# Az eredeti feature oszlopok listáját előzetesen be kell állítani
feature_names = [...]  # Az edzés során használt összes oszlop neve

# Készítsünk egy DataFrame-et a megfelelő oszlopokkal
input_data_filled = pd.DataFrame(columns=feature_names)
input_data_filled.loc[0, ['results', 'year', 'month']] = [results_val, year, month]

# Hiányzó oszlopokat 0-ra állítjuk
input_data_filled.fillna(0, inplace=True)

# Standardizálás
input_data_scaled = scaler.transform(input_data_filled)

# Biztosítsuk, hogy az alak megfeleljen a hálózat elvárásainak
input_data_scaled = np.expand_dims(input_data_scaled, axis=0)  # (1, 565)

if st.button("Előrejelzés"):
    if model_choice == "Random Forest":
        probs = rf_model.predict_proba(input_data_scaled)[0]
    else:
        probs = nn_model.predict(input_data_scaled)[0]

    # Top 3 legvalószínűbb eredmény visszaalakítása
    top_3_indices = np.argsort(probs)[-3:][::-1]

    # Ellenőrizzük, hogy az előrejelzési eredmények szerepelnek-e az eredeti osztályok között
    valid_indices = [i for i in top_3_indices if i in le.classes_]
    if valid_indices:
        top_3_tournaments = le.inverse_transform(valid_indices)
    else:
        top_3_tournaments = ["Ismeretlen" for _ in range(len(top_3_indices))]

    top_3_probs = [probs[i] for i in top_3_indices]

    st.markdown("### Előrejelzések:")
    for i in range(3):
        st.write(f"**{top_3_tournaments[i]}:** {top_3_probs[i]:.2f} valószínűséggel")
