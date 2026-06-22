import streamlit as st
import pickle
import numpy as np
import pandas as pd
model = pickle.load(open("disease_model.pkl", "rb"))
symptom_list = pickle.load(open("symptom_list.pkl", "rb"))
description_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("Disease Predictor Based on Symptoms")
st.write("Select your symptoms below and click Predict to find out the possible disease.")
selected_symptoms = st.multiselect(
    "Select Symptoms:",
    options=sorted(symptom_list)
)

if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom!")
    else:
        input_data = np.zeros(len(symptom_list))
        for symptom in selected_symptoms:
            if symptom in symptom_list:
                index = symptom_list.index(symptom)
                input_data[index] = 1
        input_df = pd.DataFrame([input_data], columns=symptom_list)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)
        confidence = round(max(probability[0]) * 100, 2)
        st.success(f"Predicted Disease: {prediction}")
        st.info(f"Confidence: {confidence}%")
        desc = description_df[description_df["Disease"] == prediction]["Description"].values
        if len(desc) > 0:
            st.subheader("About this Disease:")
            st.write(desc[0])

        prec = precaution_df[precaution_df["Disease"] == prediction]
        if len(prec) > 0:
            st.subheader("Precautions:")
            for i in range(1, 5):
                val = prec[f"Precaution_{i}"].values[0]
                if pd.notna(val):
                    st.write(f"{i}. {val}")
        st.error("Always consult a real doctor for proper diagnosis!")