import pandas as pd
import mlflow
# Tidak perlu mlflow.sklearn.autolog() di sini karena kita akan manual log
from sklearn.model_selection import train_test_split, GridSearchCV # Import GridSearchCV untuk tuning
from sklearn.ensemble import RandomForestClassifier # Contoh model untuk tuning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import logging
import os
import json # Untuk menyimpan classification report sebagai JSON
import matplotlib.pyplot as plt # Untuk plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay # Untuk plot confusion matrix
import numpy as np # Untuk operasi array

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model_tuned(data_path):
    """
    Melatih model RandomForestClassifier dengan hyperparameter tuning
    dan MLflow manual logging.
    """
    logging.info(f"Memulai tuning model dengan manual logging untuk data: {data_path}")

    # Muat data yang sudah diproses
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data berhasil dimuat. Bentuk: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: File data yang diproses tidak ditemukan di {data_path}")
        return

    # Pisahkan fitur (X) dan target (y)
    X = df.drop('Personality', axis=1)
    y = df['Personality']
    logging.info("Fitur dan target berhasil dipisahkan.")

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"Data dibagi menjadi set pelatihan ({X_train.shape[0]} sampel) dan pengujian ({X_test.shape[0]} sampel).")

    # Definisikan model dasar
    base_model = RandomForestClassifier(random_state=42)

    # Definisikan grid hyperparameter untuk tuning
    param_grid = {
        'n_estimators': [50, 100], # Jumlah pohon dalam Random Forest
        'max_depth': [None, 10, 20],   # Kedalaman maksimum pohon
        'min_samples_split': [2, 5],   # Minimum sampel untuk split
    }
    logging.info(f"Grid hyperparameter didefinisikan: {param_grid}")

    # Gunakan GridSearchCV untuk mencari hyperparameter terbaik
    # cv=3 berarti 3-fold cross-validation
    # scoring='f1_weighted' adalah metrik yang digunakan untuk memilih model terbaik selama CV
    # n_jobs=-1 akan menggunakan semua core CPU yang tersedia
    logging.info("Memulai GridSearchCV...")
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Ambil model terbaik dari GridSearch
    best_model = grid_search.best_estimator_
    logging.info(f"Model terbaik ditemukan: {best_model}")
    logging.info(f"Hyperparameter terbaik: {grid_search.best_params_}")
    logging.info(f"Skor F1 terbaik dari Cross-Validation: {grid_search.best_score_:.4f}")

    # --- MLflow Manual Logging ---
    # Mulai MLflow Run
    with mlflow.start_run(run_name="Random_Forest_Tuning_Manual_Log"):
        # Log parameter terbaik secara manual
        mlflow.log_params(grid_search.best_params_)
        logging.info("Hyperparameter terbaik dicatat secara manual ke MLflow.")

        # Prediksi pada data uji dengan model terbaik
        y_pred = best_model.predict(X_test)

        # Hitung dan catat metrik evaluasi secara manual (metrik yang sama dengan autolog)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        logging.info(f"Metrik pengujian dicatat secara manual: Akurasi={accuracy:.4f}, F1-Score={f1:.4f}")

        # Catat classification report sebagai artefak tambahan (opsional tapi disarankan)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_filepath = "classification_report.json"
        with open(report_filepath, "w") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact(report_filepath)
        os.remove(report_filepath) # Hapus file lokal setelah diunggah
        logging.info("Laporan klasifikasi dicatat sebagai artefak.")
        
        # Simpan model terbaik sebagai artefak secara manual
        mlflow.sklearn.log_model(best_model, "best_random_forest_model_tuned")
        logging.info("Model terbaik yang telah di-tuning dicatat sebagai artefak.")

    logging.info("MLflow run untuk model yang di-tuning selesai.")

if __name__ == "__main__":
    # Path ke data yang sudah diproses (relatif dari lokasi skrip modelling.py)
    PROCESSED_DATA_PATH = 'personality_preprocessing/processed_data.csv' # Hapus `../` di awal jika ada
    train_model_tuned(PROCESSED_DATA_PATH)
