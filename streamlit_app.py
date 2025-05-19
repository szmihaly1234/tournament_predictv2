import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Modellek és skalázó betöltése
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
nn_model = tf.keras.models.load_model("nn_model.h5")

st.title("Match Outcome Prediction")

st.markdown("""
Ez az alkalmazás két különböző modellt használ a mérkőzés eredményének előrejelzésére:
- **Random Forest**
- **Neurális Hálózat**

Add meg a mérkőzés adatait az alábbiak szerint:
""")

# Az eredmény oszlop értékeinek kiválasztása (figyelem: a LabelEncoder által képzett értékek!)
results_val = st.selectbox("Eredmény (pl. 0 = 'draw', 1 = 'home_win', 2 = 'away_win')", [0, 1, 2])
year_val = st.number_input("Év", value=2020, step=1)
month_val = st.number_input("Hónap (1-12)", value=1, step=1, min_value=1, max_value=12)

# Az input vektor elkészítése – ugyanolyan sorrendben, mint ahogy a modellek tréning során használtad
input_data = np.array([[results_val, year_val, month_val]])

# Standardizáljuk az input adatok értékeit
input_data_scaled = scaler.transform(input_data)

if st.button("Előrejelzés"):
    # Random Forest előrejelzés
    rf_pred = rf_model.predict(input_data_scaled)
    
    # Neurális hálózat előrejelzés: softmax kimenet, ezért az osztályindex az argmax
    nn_prob = nn_model.predict(input_data_scaled)
    nn_pred = np.argmax(nn_prob, axis=1)
    
    st.markdown("### Előrejelzések:")
    st.write("**Random Forest** modell előrejelzése:", rf_pred[0])
    st.write("**Neurális Hálózat** modell előrejelzése:", nn_pred[0])
