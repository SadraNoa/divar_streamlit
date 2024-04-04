import streamlit as st
import numpy as np
import pandas as pd
import joblib

loaded_model = joblib.load('divar.joblib')

st.title('Tehran houses Prediction in 2024')
st.write("""### We need some information to predict the """)

address = (
       'Shahran', 'Pardis', 'Shahrake Qods', 'Shahrake Gharb',
       'North Program Organization', 'Andisheh', 'West Ferdows Boulevard',
       'Narmak', 'Saadat Abad', 'Zafar', 'Islamshahr', 'Pirouzi',
       'Shahrake Shahid Bagheri', 'Moniriyeh', 'Velenjak', 'Amirieh',
       'Southern Janatabad', 'Salsabil', 'Zargandeh', 'Feiz Garden',
       'Water Organization', 'ShahrAra', 'Gisha', 'Ray', 'Abbasabad',
       'Ostad Moein', 'Farmanieh', 'Parand', 'Punak', 'Qasr-od-Dasht',
       'Aqdasieh', 'Pakdasht', 'Railway', 'Central Janatabad',
       'East Ferdows Boulevard', 'Pakdasht KhatunAbad', 'Sattarkhan',
       'Baghestan', 'Shahryar', 'Northern Janatabad', 'Daryan No',
       'Southern Program Organization', 'Rudhen', 'West Pars', 'Afsarieh',
       'Marzdaran', 'Dorous', 'Sadeghieh', 'Chahardangeh', 'Baqershahr',
       'Jeyhoon', 'Lavizan', 'Shams Abad', 'Fatemi',
       'Keshavarz Boulevard', 'Kahrizak', 'Qarchak',
       'Northren Jamalzadeh', 'Azarbaijan', 'Bahar',
       'Persian Gulf Martyrs Lake', 'Beryanak', 'Heshmatieh',
       'Elm-o-Sanat', 'Golestan', 'Shahr-e-Ziba', 'Pasdaran',
       'Chardivari', 'Gheitarieh', 'Kamranieh', 'Gholhak', 'Heravi',
       'Hashemi', 'Dehkade Olampic', 'Damavand', 'Republic', 'Zaferanieh',
       'Qazvin Imamzadeh Hassan', 'Niavaran', 'Valiasr', 'Qalandari',
       'Amir Bahador', 'Ekhtiarieh', 'Ekbatan', 'Absard', 'Haft Tir',
       'Mahallati', 'Ozgol', 'Tajrish', 'Abazar', 'Koohsar', 'Hekmat',
       'Parastar', 'Lavasan', 'Majidieh', 'Southern Chitgar', 'Karimkhan',
       'Si Metri Ji', 'Karoon', 'Northern Chitgar', 'East Pars', 'Kook',
       'Air force', 'Sohanak', 'Komeil', 'Azadshahr', 'Zibadasht',
       'Amirabad', 'Dezashib', 'Elahieh', 'Mirdamad', 'Razi', 'Jordan',
       'Mahmoudieh', 'Shahedshahr', 'Yaftabad', 'Mehran', 'Nasim Shahr',
       'Tenant', 'Chardangeh', 'Fallah', 'Eskandari', 'Shahrakeh Naft',
       'Ajudaniye', 'Tehransar', 'Nawab', 'Yousef Abad',
       'Northern Suhrawardi', 'Villa', 'Hakimiyeh', 'Nezamabad',
       'Garden of Saba', 'Tarasht', 'Azari', 'Shahrake Apadana', 'Araj',
       'Vahidieh', 'Malard', 'Shahrake Azadi', 'Darband', 'Vanak',
       'Tehran Now', 'Darabad', 'Eram', 'Atabak', 'Sabalan', 'SabaShahr',
       'Shahrake Madaen', 'Waterfall', 'Ahang', 'Salehabad', 'Pishva',
       'Enghelab', 'Islamshahr Elahieh', 'Ray - Montazeri',
       'Firoozkooh Kuhsar', 'Ghoba', 'Mehrabad', 'Southern Suhrawardi',
       'Abuzar', 'Dolatabad', 'Hor Square', 'Taslihat', 'Kazemabad',
       'Robat Karim', 'Ray - Pilgosh', 'Ghiyamdasht', 'Telecommunication',
       'Mirza Shirazi', 'Gandhi', 'Argentina', 'Seyed Khandan',
       'Shahrake Quds', 'Safadasht', 'Khademabad Garden', 'Hassan Abad',
       'Chidz', 'Khavaran', 'Boloorsazi', 'Mehrabad River River',
       'Varamin - Beheshti', 'Shoosh', 'Thirteen November', 'Darakeh',
       'Aliabad South', 'Alborz Complex', 'Firoozkooh', 'Vahidiyeh',
       'Shadabad', 'Naziabad', 'Javadiyeh', 'Yakhchiabad'
)

area = st.number_input("Enter the area: ", 1, 200)
room = st.slider("Enter the number of room: ", 0, 5, 2)
parking = st.select_slider("Choose parking", ['Yes', 'No'])
warehouse = st.select_slider("Choose warehouse", ['Yes', 'No'])
elevator = st.select_slider("Choose elevator", ['Yes', 'No'])
selected_address = st.selectbox("Address", address)

parking = 1 if parking == 'Yes' else 0
warehouse = 1 if warehouse == 'Yes' else 0
elevator = 1 if elevator == 'Yes' else 0

def predict(area, room, parking, warehouse, elevator, selected_address):
    row = np.array([area, room, parking, warehouse, elevator, selected_address])
    X = pd.DataFrame([row], columns=['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address'])
    prediction = loaded_model.predict(X)
    return prediction

trigger = st.button('Predict')

if trigger:
    prediction_result = predict(area, room, parking, warehouse, elevator, selected_address)
    st.write("Prediction Result:", prediction_result)

