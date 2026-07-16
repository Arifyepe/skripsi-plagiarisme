import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Data
df = pd.read_csv("dataset_skripsi_final.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates(subset=['text'])
X = df["text"].astype(str)
y = df["label"].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load Models
vectorizer = joblib.load("vectorizer.joblib")
scaler = joblib.load("scaler.joblib")
svm_model = joblib.load("svm_model.joblib")

# Preprocess
X_test_tfidf = vectorizer.transform(X_test)

import string
def get_stylometry(text):
    words = text.split()
    word_count = len(words) if len(words) > 0 else 1
    char_count = len(text)
    avg_word_len = char_count / word_count
    punct_count = sum([1 for char in text if char in string.punctuation])
    return [word_count, char_count, avg_word_len, punct_count]

X_test_stylo = np.array([get_stylometry(text) for text in X_test])
X_test_stylo_scaled = scaler.transform(X_test_stylo)

from scipy.sparse import hstack
X_test_combined = hstack([X_test_tfidf, X_test_stylo_scaled])

# Predict
y_pred = svm_model.predict(X_test_combined)

# Generate Report
report_dict = classification_report(y_test, y_pred, target_names=["Manusia", "AI"], output_dict=True)
report_df = pd.DataFrame(report_dict).T

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred) * 100

plot_data = report_df.iloc[:-1, :-1]
plt.figure(figsize=(9, 5))
sns.heatmap(plot_data, annot=True, cmap="Blues", fmt=".3f", cbar=True,
            annot_kws={"size": 13}, vmin=0.96, vmax=1.00,
            linewidths=0.5, linecolor='white')
plt.title(f"Classification Report Heatmap (Akurasi {acc:.2f}%)", pad=20, fontsize=16)
plt.ylabel("Kelas/Metrik")
plt.xlabel("Skor")
plt.tight_layout()
plt.savefig("classification_report_svm.png", dpi=300)
print("Classification report saved to classification_report_svm.png")

