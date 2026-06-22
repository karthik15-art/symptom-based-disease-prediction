import pickle
import numpy as np
import pandas as pd
model = pickle.load(open("disease_model.pkl", "rb"))
symptom_list = pickle.load(open("symptom_list.pkl", "rb"))

def predict_disease(selected_symptoms):
    input_data = np.zeros(len(symptom_list))
    for symptom in selected_symptoms:
        if symptom in symptom_list:
            index = symptom_list.index(symptom)
            input_data[index] = 1


    input_df = pd.DataFrame([input_data], columns=symptom_list)
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    confidence = round(max(probability[0]) * 100, 2)
    return prediction[0], confidence
test_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
disease, confidence = predict_disease(test_symptoms)
print("Predicted Disease:", disease)
print("Confidence:", confidence, "%")