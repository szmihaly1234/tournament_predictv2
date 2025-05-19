import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io  # in-memory fájlkezeléshez
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Alap Streamlit beállítások
st.set_page_config(page_title="Futball mérkőzés előrejelzés", layout="wide")
st.title("Futball mérkőzés előrejelzés")
st.write("Az alkalmazás az `all_matches.csv` fájlból dolgozik, amely a repository-ban található.")

# 1. Adatok betöltése a repository-ból (feltételezzük, hogy az app.py mellett van az all_matches.csv)
@st.cache_data
def load_data():
    df = pd.read_csv("all_matches.csv")
    return df

df = load_data()

st.subheader("Feltöltött adatok első néhány sora")
st.dataframe(df.head())
st.write("Sorok száma:", df.shape[0])
st.write("Oszlopok száma:", df.shape[1])
st.write("Tournament-ek száma:", df['tournament'].nunique())

# 2. Adattisztítás és feature engineering
# Töröljük a felesleges oszlopokat, ha léteznek
for col in ['neutral', 'country']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Győzelmek meghatározása: ha a hazai több gólt szerzett → hazai győzelem, kevesebb esetén idegen, egyenlőnél döntetlen
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)
df['results'] = np.where(df['home_score'] > df['away_score'], 'home_win',
                         np.where(df['home_score'] < df['away_score'], 'away_win', 'draw'))
df.drop(['home_score', 'away_score'], axis=1, inplace=True)

# Dátum konvertálása és extra oszlopok (év, hónap, nap)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Ha egy tournamentnek kevesebb mint 100 meccse van, cseréljük "other"-re
df['tournament'] = np.where(df['tournament'].map(df['tournament'].value_counts()) < 100,
                            'other', df['tournament'])

# Label encoding: két külön enkóder a tournament és az eredmények számára
tournament_le = LabelEncoder()
results_le = LabelEncoder()
df['tournament'] = tournament_le.fit_transform(df['tournament'])
df['results'] = results_le.fit_transform(df['results'])

# Csapatok enkódolása: egyesítsük a hazai és idegen csapatokat, hogy ugyanaz a numerikus kód kerüljön minden előfordulásnál
teams = pd.concat([df['home_team'], df['away_team']])
teams_unique = teams.unique()
team_le = LabelEncoder()
team_le.fit(teams_unique)
df['home_team_encoded'] = team_le.transform(df['home_team'])
df['away_team_encoded'] = team_le.transform(df['away_team'])

# 3. Felhasználói beviteli adatok kiválasztása a sidebar-on
teams_list = sorted(team_le.classes_)
st.sidebar.header("Mérkőzés adatai")
home_team_choice = st.sidebar.selectbox("Válaszd ki a hazai csapatot", teams_list)
away_team_choice = st.sidebar.selectbox("Válaszd ki az idegen csapatot", teams_list)

match_date = st.sidebar.date_input("Mérkőzés dátuma", datetime.date.today())
year_input = match_date.year
month_input = match_date.month
day_input = match_date.day

# Mérkőzés eredménye: jelenítsük meg a lehetőségeket magyarul, majd mappeljük az angol kódokra
result_mapping = {
    "Hazai győzelem": "home_win",
    "Idegen győzelem": "away_win",
    "Döntetlen": "draw"
}
result_choice = st.sidebar.selectbox("Válaszd ki a mérkőzés eredményét", list(result_mapping.keys()))
result_str = result_mapping[result_choice]
result_encoded = int(results_le.transform([result_str])[0])

home_team_encoded_input = int(team_le.transform([home_team_choice])[0])
away_team_encoded_input = int(team_le.transform([away_team_choice])[0])

# Az alkalmazandó feature-ök: results, year, month, day, home_team_encoded, away_team_encoded
x_new = np.array([[result_encoded, year_input, month_input, day_input,
                     home_team_encoded_input, away_team_encoded_input]])

# 4. Feature kiválasztás és előkészítés a modell tanításhoz
features = ['results', 'year', 'month', 'day', 'home_team_encoded', 'away_team_encoded']
X = df[features]
y = df['tournament']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_new_scaled = scaler.transform(x_new)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Modell választás
st.sidebar.header("Modell választása")
model_type = st.sidebar.radio("Válaszd ki a modellt", ("Random Forest", "Neurális Háló"))

if st.sidebar.button("Előrejelzés és modell mentése"):
    if model_type == "Random Forest":
        st.write("Random Forest modell tanítása folyamatban...")
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)
        pred = model_rf.predict(x_new_scaled)
        predicted_tournament = tournament_le.inverse_transform(pred)[0]
        st.success(f"A becsült tournament: **{predicted_tournament}**")
        
        # Mentés: a pickle-be csomagoljuk a modellt, és letöltésre ajánljuk
        buffer = io.BytesIO()
        pickle.dump(model_rf, buffer)
        buffer.seek(0)
        st.download_button(label="Random Forest modell letöltése",
                           data=buffer,
                           file_name="model_rf.pkl",
                           mime="application/octet-stream")
        
    else:
        st.write("Neurális háló tanítása folyamatban (kérlek várj egy kicsit)...")
        nn_model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(len(np.unique(y)), activation='softmax')
        ])
        nn_model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        nn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        pred_prob = nn_model.predict(x_new_scaled)
        pred = [np.argmax(pred_prob, axis=1)[0]]
        predicted_tournament = tournament_le.inverse_transform(pred)[0]
        st.success(f"A becsült tournament: **{predicted_tournament}**")
        
        # H5 fájlba mentjük a neurális hálót
        nn_model.save("model_nn.h5")
        # Olvassuk be a fájlt byte formátumban, hogy download_button használhassa
        with open("model_nn.h5", "rb") as f:
            nn_bytes = f.read()
        st.download_button(label="Neurális Háló modell letöltése",
                           data=nn_bytes,
                           file_name="model_nn.h5",
                           mime="application/octet-stream")
