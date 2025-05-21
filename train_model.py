import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Wczytanie danych
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv(url, names=columns)

# Podział danych
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenujemy model drzewa z ograniczeniami (lepsze uogólnienie)
model = DecisionTreeClassifier(
    max_depth=5,            # ograniczenie głębokości drzewa
    min_samples_split=10,   # minimalna liczba próbek do podziału
    min_samples_leaf=5,     # minimalna liczba próbek w liściu
    random_state=42
)
model.fit(X_train, y_train)

# Ocena modelu
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Dokładność na zbiorze testowym: {acc:.2%}")
print(classification_report(y_test, y_pred))

# Zapisz model
joblib.dump(model, "model.pkl")
print("Model zapisany jako model.pkl")
