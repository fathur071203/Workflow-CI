import pandas as pd
import mlflow
from mlflow.exceptions import MlflowException
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, log_loss 
import logging
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model_tuned(data_path):
    logging.info(f"Memulai tuning model dengan manual logging untuk data: {data_path}")

    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data berhasil dimuat. Bentuk: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: File data yang diproses tidak ditemukan di {data_path}")
        return

    X = df.drop('Personality', axis=1)
    y = df['Personality']
    logging.info("Fitur dan target berhasil dipisahkan.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"Data dibagi menjadi set pelatihan ({X_train.shape[0]} sampel) dan pengujian ({X_test.shape[0]} sampel).")

    base_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100], 
        'max_depth': [None, 10, 20],   
        'min_samples_split': [2, 5],   
    }
    logging.info(f"Grid hyperparameter didefinisikan: {param_grid}")

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

    mlflow.set_tracking_uri("./mlruns") 

    run_id_from_env = os.environ.get("MLFLOW_RUN_ID")
    if not run_id_from_env:
        logging.error("MLFLOW_RUN_ID tidak ditemukan di environment. Tidak dapat melog ke run yang ada.")
        exit(1)

    try:
        with mlflow.start_run(run_id=run_id_from_env, experiment_id="0", nested=True) as run: 
            logging.info(f"Melakukan logging metrik dan artefak ke MLflow run yang aktif (ID: {run.info.run_id}).")
            
            mlflow.log_params(grid_search.best_params_)
            logging.info("Hyperparameter terbaik dicatat secara manual ke MLflow.")

            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test) 

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)
            logging.info(f"Metrik pengujian dicatat: Akurasi={accuracy:.4f}, F1-Score={f1:.4f}")

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

            mlflow.sklearn.log_model(best_model, "model") 
            logging.info("Model terbaik yang telah di-tuning dicatat sebagai artefak.")

            print(f"MLFLOW_FINAL_RUN_ID:{run.info.run_id}")

        logging.info("--- Proses pelatihan model selesai. ---")

    except MlflowException as mlflow_e:
        logging.error(f"Terjadi kesalahan MLflow yang tidak terduga: {mlflow_e}")
        exit(1)
    except Exception as e:
        logging.error(f"Terjadi kesalahan umum saat melatih model: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    # --- UBAH PATH INI! ---
    # Path ini sekarang harus relatif dari ROOT repositori Workflow-CI,
    # karena skrip dijalankan dari sana.
    PROCESSED_DATA_PATH = 'MLProject/personality_preprocessing/processed_data.csv' 
    train_model_tuned(PROCESSED_DATA_PATH)
