import pandas as pd
import mlflow
# Tidak perlu mlflow.sklearn.autolog() di sini karena kita akan manual log
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report # Untuk Kriteria 2 Skilled
# Jika Anda mengaktifkan metrik tambahan di Kriteria 2 Advanced, tambahkan import di bawah
# from sklearn.metrics import roc_auc_score, log_loss 
import logging
import os
import json # Untuk menyimpan classification report sebagai JSON
import matplotlib.pyplot as plt # Untuk plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay # Untuk plot confusion matrix
import numpy as np # Untuk operasi array
import time # Untuk mencatat waktu pelatihan (opsional)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model_tuned(data_path): # Ini akan dijalankan oleh MLflow Project
    """
    Melatih model RandomForestClassifier dengan hyperparameter tuning
    dan MLflow manual logging. Ini akan dijalankan oleh MLflow Project.
    """
    # --- PENTING: Bagian yang DIHAPUS karena MLflow Project yang mengelola run ---
    # if 'MLFLOW_RUN_ID' in os.environ:
    #     logging.info("MLFLOW_RUN_ID ditemukan di lingkungan, menghapusnya untuk memastikan run baru.")
    #     del os.environ['MLFLOW_RUN_ID']
    # --- AKHIR Bagian yang DIHAPUS ---

    logging.info(f"Memulai tuning model dengan manual logging untuk data: {data_path}")

    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data berhasil dimuat. Bentuk: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: File data yang diproses tidak ditemukan di {data_path}")
        return

    # Pisahkan fitur (X) dan target (y)
    X = df.drop('Personality', axis=1) # Asumsi 'Personality' adalah kolom target
    y = df['Personality']
    logging.info("Fitur dan target berhasil dipisahkan.")

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"Data dibagi menjadi set pelatihan ({X_train.shape[0]} sampel) dan pengujian ({X_test.shape[0]} sampel).")

    # Definisikan model dasar
    base_model = RandomForestClassifier(random_state=42)

    # Definisikan grid hyperparameter untuk tuning
    param_grid = {
        'n_estimators': [50, 100], 
        'max_depth': [None, 10, 20],   
        'min_samples_split': [2, 5],   
    }
    logging.info(f"Grid hyperparameter didefinisikan: {param_grid}")

    # Gunakan GridSearchCV untuk mencari hyperparameter terbaik
    logging.info("Memulai GridSearchCV...")
    start_time_training = time.time() 
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    end_time_training = time.time() 
    training_time = end_time_training - start_time_training
    logging.info(f"Pelatihan dan tuning model selesai dalam {training_time:.2f} detik.")

    best_model = grid_search.best_estimator_
    logging.info(f"Model terbaik ditemukan: {best_model}")
    logging.info(f"Hyperparameter terbaik: {grid_search.best_params_}")
    logging.info(f"Skor F1 terbaik dari Cross-Validation: {grid_search.best_score_:.4f}")

    # --- MLflow Manual Logging ---
    # PENTING: TIDAK ADA with mlflow.start_run() DI SINI. MLflow Project sudah membuat run yang aktif.
    # Setel MLflow Tracking URI agar logging terjadi di tempat yang benar
    mlflow.set_tracking_uri("./mlruns") # Pastikan MLflow melacak ke folder lokal

    # PENTING: Dapatkan run aktif yang dibuat oleh MLflow Project
    try:
        current_run = mlflow.active_run()
        if current_run is None:
            # Fallback jika tidak ada run aktif (jarang terjadi di konteks MLProject)
            logging.warning("Tidak ada run MLflow aktif, memulai run baru secara manual.")
            current_run = mlflow.start_run(run_name="Fallback_CI_Run")
            
    except Exception as e:
        logging.error(f"Gagal mendapatkan/memulai run MLflow: {e}")
        # Ini menandakan masalah serius, mungkin perlu exit
        exit(1)


    with current_run: # Gunakan run yang aktif atau yang baru dibuat sebagai konteks
        logging.info("Melakukan logging metrik dan artefak ke MLflow run yang aktif (dari MLflow Project).")
        
        # Log parameter terbaik secara manual
        mlflow.log_params(grid_search.best_params_)
        logging.info("Hyperparameter terbaik dicatat secara manual ke MLflow.")

        # Prediksi pada data uji dengan model terbaik
        y_pred = best_model.predict(X_test)
        # y_pred_proba dibutuhkan untuk ROC AUC dan Log Loss, yang ada di Kriteria 2 Advance.
        y_pred_proba = best_model.predict_proba(X_test) 

        # Hitung dan catat metrik evaluasi dasar (sesuai Kriteria 2 Skilled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        logging.info(f"Metrik pengujian dicatat: Akurasi={accuracy:.4f}, F1-Score={f1:.4f}")

        # --- Metrik Tambahan (Contoh dari Kriteria 2 Advanced, jika Anda ingin menyertakannya) ---
        try:
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1]) 
                mlflow.log_metric("test_roc_auc_score", roc_auc)
                logging.info(f"Metrik tambahan - ROC AUC: {roc_auc:.4f}")
            else:
                logging.warning("ROC AUC tidak dihitung: Probabilitas prediksi tidak 2D atau hanya satu kelas.")
        except ValueError as e:
            logging.warning(f"Tidak dapat menghitung ROC AUC: {e}. Periksa label target atau probabilitas.")

        try:
            logloss = log_loss(y_test, y_pred_proba)
            mlflow.log_metric("test_log_loss", logloss)
            logging.info(f"Metrik tambahan - Log Loss: {logloss:.4f}")
        except ValueError as e:
            logging.warning(f"Tidak dapat menghitung Log Loss: {e}. Periksa label target atau probabilitas.")

        mlflow.log_metric("training_time_seconds", training_time)
        logging.info(f"Metrik tambahan - Waktu Pelatihan: {training_time:.2f}s")
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
            feature_importance_dict = {
                X.columns[i]: imp for i, imp in enumerate(feature_importances)
            }
            mlflow.log_dict(feature_importance_dict, "feature_importances.json")
            logging.info("Pentingnya fitur dicatat sebagai artefak kamus.")


        # --- Log Artefak Tambahan ---
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_filepath = "classification_report.json"
        with open(report_filepath, "w") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact(report_filepath)
        os.remove(report_filepath) 
        logging.info("Laporan klasifikasi dicatat sebagai artefak.")

        cm_display = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=["Introvert", "Extrovert"]) 
        plt.figure(figsize=(8, 6)) 
        cm_display.plot(cmap=plt.cm.Blues, values_format='d') 
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_img_path = "confusion_matrix.png"
        plt.savefig(cm_img_path)
        mlflow.log_artifact(cm_img_path)
        plt.close() 
        os.remove(cm_img_path)
        logging.info("Plot confusion matrix dicatat sebagai artefak.")

        mlflow.sklearn.log_model(best_model, "model") # Model akan disimpan di folder 'model' di dalam artifacts
        logging.info("Model terbaik yang telah di-tuning dicatat sebagai artefak.")

        # --- PENTING: Cetak Run ID untuk diambil oleh GitHub Actions ---
        print(f"MLFLOW_FINAL_RUN_ID:{mlflow.active_run().info.run_id}")

    logging.info("--- Proses pelatihan model selesai. ---")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = 'personality_preprocessing/processed_data.csv' # Path relatif dari modelling.py
    train_model_tuned(PROCESSED_DATA_PATH)
