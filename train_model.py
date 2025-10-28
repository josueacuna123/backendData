import os
import io
import pickle
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from supabase import create_client
from dotenv import load_dotenv
import joblib


# ===============================
# 🌍 Configuración de Supabase
# ===============================
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("❌ Faltan SUPABASE_URL o SUPABASE_KEY en el archivo .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===============================
# ⚙️ Función auxiliar de logs
# ===============================
def send_log(message: str, websocket=None):
    print(message)
    if websocket:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(websocket.send_text(message))
        except RuntimeError:
            asyncio.run(websocket.send_text(message))

# ===============================
# 🧠 Función principal de entrenamiento
# ===============================
def train_model(dataset_id: str, model_type: str, target_column: str, websocket=None):
    try:
        send_log("📦 Cargando dataset desde Supabase...", websocket)
        response = supabase.table("dataset_rows").select("data").eq("dataset_id", dataset_id).execute()

        if not response.data or len(response.data) == 0:
            raise Exception("No se encontraron datos para este dataset")

        df = pd.DataFrame([row["data"] for row in response.data])
        send_log(f"✅ Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas", websocket)

        if target_column not in df.columns:
            raise Exception(f"La columna objetivo '{target_column}' no existe en el dataset")

        # ===============================
        # 🧹 Preprocesamiento
        # ===============================
        send_log("⚙️ Preprocesando datos...", websocket)
        df = df.dropna(subset=[target_column])

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 🔍 Detección robusta de tipo de tarea
        try:
            y_num = pd.to_numeric(y, errors="coerce")
            if y_num.notnull().all():
                y = y_num
                task_type = "regression"
            else:
                task_type = "classification"
        except Exception:
            task_type = "classification"

        send_log(f"🧩 Tipo de tarea detectado: {task_type}", websocket)

        # Codificar columnas categóricas
        for col in X.columns:
            if X[col].dtype == "object":
                send_log(f"🔄 Codificando columna categórica: {col}", websocket)
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Codificar target si es clasificación
        if task_type == "classification":
            send_log(f"🎯 Codificando target: {target_column}", websocket)
            y = LabelEncoder().fit_transform(y.astype(str))

        feature_names = X.columns.tolist()
        
        # Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)    # ✅ Escala solo las features


        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ===============================
        # 🧠 Entrenamiento
        # ===============================
        send_log(f"🧠 Entrenando modelo {model_type} ({task_type})...", websocket)

        if model_type == "RandomForest":
            model = (
                RandomForestClassifier(n_estimators=100, random_state=42)
                if task_type == "classification"
                else RandomForestRegressor(n_estimators=100, random_state=42)
            )
        elif model_type == "GradientBoosting":
            model = (
                GradientBoostingClassifier(random_state=42)
                if task_type == "classification"
                else GradientBoostingRegressor(random_state=42)
            )
        elif model_type == "NeuralNetwork":
            model = (
                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
                if task_type == "classification"
                else MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
            )
        else:
            raise Exception(f"Modelo no soportado: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        
        # ===============================
        # 📊 Evaluación
        # ===============================
        if task_type == "classification":
            score = float(accuracy_score(y_test, y_pred))
            send_log(f"✅ Precisión del modelo (accuracy): {score:.4f}", websocket)
        else:
            score = float(r2_score(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            send_log(f"✅ R² del modelo: {score:.4f}", websocket)
            send_log(f"📉 Error cuadrático medio (MSE): {mse:.2f}", websocket)

        # ===============================
        # 💾 Guardar modelo entrenado
        # ===============================
        send_log("💾 Guardando modelo entrenado en Supabase...", websocket)

# Guardar todos los componentes importantes
        model_package = {
            "model": model,
            "scaler": scaler,
            "encoders": {},
            "features": feature_names,      # ✅ Guardamos las columnas usadas para el entrenamiento
            "target": target_column         # ✅ También el nombre del target
        }

        # Guardar los LabelEncoders usados
        for col in df.columns:
            if df[col].dtype == "object":
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                model_package["encoders"][col] = le

        buffer = io.BytesIO()
        pickle.dump(model_package, buffer)
        buffer.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{task_type}_{dataset_id[:6]}_{target_column}_{timestamp}"
        file_name = f"{model_name}.pkl"
        bucket_name = "models"

        res = None  # ✅ Previene el warning de Pylance

        try:
            supabase.storage.from_(bucket_name).upload(file_name, buffer.getvalue(), {"upsert": True})
        except Exception:
            send_log("⚠️ SDK antiguo detectado, usando método alternativo...", websocket)
            import requests
            storage_url = f"{SUPABASE_URL}/storage/v1/object/{bucket_name}/{file_name}?upsert=true"
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/octet-stream",
            }
            res = requests.post(storage_url, headers=headers, data=buffer.getvalue())
            if res.status_code not in (200, 201):
                raise Exception(f"Error al subir el modelo: {res.text}")

        public_url = supabase.storage.from_(bucket_name).get_public_url(file_name)
        send_log(f"🌐 Modelo subido correctamente al bucket '{bucket_name}': {public_url}", websocket)
        # ===============================
        # 🧾 Guardar metadatos
        # ===============================
        supabase.table("models").insert({
            "dataset_id": dataset_id,
            "model_name": model_name,
            "algorithm": model_type,
            "task_type": task_type,
            "accuracy": score,
            "trained_at": datetime.now().isoformat(),
            "target_column": target_column,
            "model_url": public_url,
            "features": feature_names  # ✅ Metadato de respaldo
        }).execute()

        send_log("🎉 Entrenamiento completado y modelo guardado correctamente.", websocket)
        return {"model_name": model_name, "accuracy": score, "url": public_url}

    except Exception as e:
        send_log(f"❌ Error durante el entrenamiento: {str(e)}", websocket)
        raise
