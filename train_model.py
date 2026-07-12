import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

print("1. Memulai proses pelatihan model...")

# ==========================================
# FUNGSI PREPROCESSING (Sama dengan di Web)
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_stylometric_features(text_series):
    features = []
    for text in text_series:
        if not isinstance(text, str):
            text = ""
        words = text.split()
        word_count = len(words) if len(words) > 0 else 1
        char_count = len(text)
        avg_word_len = char_count / word_count
        punct_count = sum([1 for char in text if char in string.punctuation])
        features.append([word_count, char_count, avg_word_len, punct_count])
    return np.array(features)

# ==========================================
# PROSES TRAINING
# ==========================================
print("2. Membaca dataset_skripsi_final.csv...")
df = pd.read_csv('dataset_skripsi_final.csv')
df.columns = df.columns.str.strip() # Membersihkan spasi pada header kolom
df = df.drop_duplicates(subset=['text']) # Hapus duplikat 

print("3. Membersihkan teks (Data Cleaning)...")
df['clean_text'] = df['text'].apply(clean_text)

# Membagi data latih dan data uji (80:20)
print("4. Melakukan Train-Test Split (80:20)...")
X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    df['clean_text'], df['label'], df['text'], test_size=0.2, random_state=42, stratify=df['label']
)

print("5. Melakukan Ekstraksi Fitur Hybrid (TF-IDF + Stylometri)...")
# TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Stylometri
scaler = StandardScaler()
X_train_stylo = extract_stylometric_features(raw_train)
X_train_stylo_scaled = scaler.fit_transform(X_train_stylo)

X_test_stylo = extract_stylometric_features(raw_test)
X_test_stylo_scaled = scaler.transform(X_test_stylo)

# Menggabungkan fitur (Hybrid)
X_train_hybrid = hstack([X_train_tfidf, X_train_stylo_scaled]).tocsr()
X_test_hybrid = hstack([X_test_tfidf, X_test_stylo_scaled]).tocsr()

y_train_array = y_train.values

# ==========================================
# EVALUASI: Stratified K-Fold Cross Validation
# ==========================================
print("\n6. Melakukan Stratified K-Fold Cross Validation (K=5) pada Data Latih...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_hybrid, y_train_array), 1):
    X_fold_train, X_fold_val = X_train_hybrid[train_idx], X_train_hybrid[val_idx]
    y_fold_train, y_fold_val = y_train_array[train_idx], y_train_array[val_idx]
    
    model_cv = SVC(kernel='linear', probability=True, random_state=42)
    model_cv.fit(X_fold_train, y_fold_train)
    
    y_pred_val = model_cv.predict(X_fold_val)
    acc = accuracy_score(y_fold_val, y_pred_val)
    fold_accuracies.append(acc)
    print(f"   - Fold {fold} Accuracy: {acc * 100:.2f}%")

print(f"Rata-rata Akurasi Validasi Silang (K-Fold CV): {np.mean(fold_accuracies) * 100:.2f}%")

print("\n7. Melatih Model Akhir (Final Model) pada Seluruh Data Latih 80%...")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_hybrid, y_train)

# Cek Akurasi Akhir pada Data Uji 20%
y_pred = svm_model.predict(X_test_hybrid)
print("\n=== HASIL EVALUASI FINAL PADA DATA UJI (20%) ===")
print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Manusia", "AI"]))

# ==========================================
# MEMBUAT GRAFIK CONFUSION MATRIX HEATMAP
# ==========================================
print("\n8. Membuat dan menyimpan grafik Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Manusia", "AI"], 
            yticklabels=["Manusia", "AI"])
plt.title("Confusion Matrix - Deteksi Teks AI (SVM)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_svm.png", dpi=300)
print("   -> Grafik berhasil disimpan sebagai 'confusion_matrix_svm.png'")

# ==========================================
# MEMBUAT GRAFIK ROC CURVE
# ==========================================
print("\n9. Membuat dan menyimpan grafik ROC Curve...")
y_pred_proba = svm_model.predict_proba(X_test_hybrid)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - SVM')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_svm.png", dpi=300)
print("   -> Grafik berhasil disimpan sebagai 'roc_curve_svm.png'")

# ==========================================
# MEMBUAT GRAFIK FEATURE IMPORTANCE (BOBOT KATA)
# ==========================================
print("10. Membuat grafik Feature Importance...")
tfidf_feature_names = vectorizer.get_feature_names_out()
stylo_feature_names = np.array(['word_count', 'char_count', 'avg_word_len', 'punct_count'])
all_feature_names = np.concatenate([tfidf_feature_names, stylo_feature_names])

coefs = svm_model.coef_.toarray()[0]
feature_importance = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coefs})

top_ai_features = feature_importance.sort_values(by='Coefficient', ascending=False).head(10)
top_manusia_features = feature_importance.sort_values(by='Coefficient', ascending=True).head(10)
top_features = pd.concat([top_manusia_features, top_ai_features])

plt.figure(figsize=(10, 8))
colors = ['blue' if c < 0 else 'red' for c in top_features['Coefficient']]
sns.barplot(x='Coefficient', y='Feature', data=top_features, palette=colors)
plt.title('Top 10 Fitur Paling Berpengaruh (Manusia vs AI)')
plt.xlabel('SVM Coefficient')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig("feature_importance_svm.png", dpi=300)
print("   -> Grafik berhasil disimpan sebagai 'feature_importance_svm.png'")

# ==========================================
# MEMBUAT GRAFIK DISTRIBUSI FITUR STILOMETRI
# ==========================================
print("11. Membuat grafik Distribusi Fitur Stilometri (Boxplot)...")
df_stylo = pd.DataFrame(extract_stylometric_features(df['text']), columns=['Word Count', 'Character Count', 'Avg Word Length', 'Punctuation Count'])
df_stylo['Label'] = df['label'].map({0: 'Manusia', 1: 'AI'})

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.boxplot(x='Label', y='Word Count', data=df_stylo, palette='Set2', hue='Label', legend=False)
plt.title('Distribusi Jumlah Kata')

plt.subplot(2, 2, 2)
sns.boxplot(x='Label', y='Character Count', data=df_stylo, palette='Set2', hue='Label', legend=False)
plt.title('Distribusi Jumlah Karakter')

plt.subplot(2, 2, 3)
sns.boxplot(x='Label', y='Avg Word Length', data=df_stylo, palette='Set2', hue='Label', legend=False)
plt.title('Rata-rata Panjang Kata')

plt.subplot(2, 2, 4)
sns.boxplot(x='Label', y='Punctuation Count', data=df_stylo, palette='Set2', hue='Label', legend=False)
plt.title('Distribusi Jumlah Tanda Baca')

plt.tight_layout()
plt.savefig("stylometry_boxplot.png", dpi=300)
print("   -> Grafik berhasil disimpan sebagai 'stylometry_boxplot.png'")

print("\n12. Menyimpan model ke dalam file .joblib...")
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(svm_model, 'svm_model.joblib')
print("SELESAI! Model dan semua grafik berhasil disimpan.")