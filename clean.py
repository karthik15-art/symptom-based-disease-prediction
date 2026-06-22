import pandas as pd
df = pd.read_csv("dataset.csv")
for col in df.columns[1:]:
    df[col] = df[col].str.strip()
symptom_columns = df.columns[1:]
all_symptoms = set()

for col in symptom_columns:
    all_symptoms.update(df[col].dropna().unique())

print("Total unique symptoms found:", len(all_symptoms))

encoded_df = pd.DataFrame(0, index=df.index, columns=list(all_symptoms))

for col in symptom_columns:
    for idx, symptom in df[col].items():
        if pd.notna(symptom):
            encoded_df.loc[idx, symptom] = 1
encoded_df.insert(0, "Disease", df["Disease"])

print("Final shape:", encoded_df.shape)
print(encoded_df.head())

 
encoded_df.to_csv("cleaned_disease_data.csv", index=False)
print("Cleaned data saved successfully!")