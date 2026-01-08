import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# loading data
data = pd.read_json("problems_data.jsonl", lines=True)
data = data.dropna(subset=["problem_class"])

# creating useful features for the model
data["problem_class"] = data["problem_class"].astype(str).str.strip().str.title()

data["combined_text"] = (
    data["title"].fillna("") + " " +
    data["description"].fillna("") + " " +
    data["input_description"].fillna("") + " " +
    data["output_description"].fillna("")
)

def extract_max_constraint(text):
    text = str(text)
    powers = re.findall(r'10\^(\d+)|10\*\*(\d+)', text)
    large_nums = re.findall(r'\b[1-9](0{3,})\b', text)

    found = []
    for p in powers:
        val = p[0] if p[0] else p[1]
        if val:
            found.append(int(val))
    for num in large_nums:
        found.append(len(num) - 1)

    return max(found) if found else 0


KEYWORDS_HARD = ["graph", "tree", "dp", "segment", "dijkstra", "modulo", "query"]
KEYWORDS_EASY = ["print", "sum", "average", "even", "odd", "brute"]
KEYWORDS_OPTIM = ["minimize", "maximize", "optimal"]

def count_keywords(text, keywords):
    text = text.lower()
    return sum(text.count(k) for k in keywords)


data["max_constraint_power"] = data["input_description"].fillna("").apply(extract_max_constraint)
data["hard_kw"] = data["combined_text"].apply(lambda x: count_keywords(x, KEYWORDS_HARD))
data["easy_kw"] = data["combined_text"].apply(lambda x: count_keywords(x, KEYWORDS_EASY))
data["optim_kw"] = data["combined_text"].apply(lambda x: count_keywords(x, KEYWORDS_OPTIM))
data["word_count"] = data["combined_text"].apply(lambda x: len(x.split()))
data["math_symbol_count"] = data["combined_text"].apply(lambda x: len(re.findall(r"[+\-*/%=<>]", x)))
data["math_density"] = data["math_symbol_count"] / data["word_count"].apply(lambda x: max(1, x))


#vectorisation
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_text = vectorizer.fit_transform(data["combined_text"])

engineered = data[
    ["word_count", "math_symbol_count", "math_density",
     "max_constraint_power", "hard_kw", "easy_kw", "optim_kw"]
].values

scaler = MaxAbsScaler()
X = hstack([X_text, scaler.fit_transform(engineered)])

le = LabelEncoder()
y_class = le.fit_transform(data["problem_class"])
y_score = data["problem_score"]


# training models
clf_rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42
)
clf_rf.fit(X, y_class)

mask = ~y_score.isna()
X_reg = X.tocsr()[mask.values]
y_reg = y_score[mask]

reg_rf = RandomForestRegressor(
    n_estimators=300, random_state=42
)
reg_rf.fit(X_reg, y_reg)


# saving the models using pickle with compression for faster loading.
joblib.dump(clf_rf, "rf_classifier.pkl", compress=3)
joblib.dump(reg_rf, "rf_regressor.pkl", compress=3)
joblib.dump(vectorizer, "tfidf_vectorizer.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl", compress=3)
joblib.dump(le, "label_encoder.pkl", compress=3)

print("Model saved successfully.")
