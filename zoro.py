import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
df = pd.read_csv("cleaned_disease_data.csv")
X = df.drop("Disease", axis=1)
y = df["Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred) * 100, "%")
pickle.dump(model, open("disease_model.pkl", "wb"))
pickle.dump(list(X.columns), open("symptom_list.pkl", "wb"))

print("Model saved successfully!")