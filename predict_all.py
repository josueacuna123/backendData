from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
import os
import io
import pandas as pd
import asyncio
from backend.train_model import train_model
from backend.routes import models_routes
from backend.predict_model import *
from backend.predict_model import *
from fastapi import APIRouter, HTTPException
import joblib
from fastapi import APIRouter
import pickle
import numpy as np


# Inicializar Supabase
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Conexión exitosa con Supabase")

def predict_all(model_url: str, dataset_id: str):
    """
    Carga un modelo desde Supabase Storage y aplica predicciones sobre
    todos los datos asociados al dataset_id especificado.
    """
    # 1️⃣ Descargar modelo desde Supabase Storage
    response = requests.get(model_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Error al descargar el modelo: {response.text}")

    try:
        model_obj = pickle.loads(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al cargar el modelo: {e}")

    # Extraer componentes del modelo
    model = model_obj.get("model")
    scaler = model_obj.get("scaler")
    encoders = model_obj.get("encoders", {})
    target_col = model_obj.get("target")  # ✅ coincide con train_model.py

    if model is None or scaler is None:
        raise HTTPException(status_code=400, detail="Modelo incompleto (faltan 'model' o 'scaler').")

    # 2️⃣ Cargar dataset completo desde Supabase
    data_response = supabase.table("dataset_rows").select("data").eq("dataset_id", dataset_id).execute()
    if not data_response.data:
        raise HTTPException(status_code=404, detail="No se encontraron datos para el dataset.")

    df = pd.DataFrame([row["data"] for row in data_response.data])

    # 3️⃣ Eliminar columna objetivo si existe
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])

    # 4️⃣ Asegurar que las features coinciden con las usadas en el entrenamiento
    feature_names = model_obj.get("features", [])
    if feature_names:
        # Solo conservar columnas que coinciden con las features del modelo
        df = df[[col for col in feature_names if col in df.columns]]

    if df.empty:
        raise HTTPException(status_code=400, detail="No hay columnas válidas para predecir en este dataset.")

    # 5️⃣ Codificar columnas categóricas
    for col in df.columns:
        if col in encoders:
            encoder = encoders[col]
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except Exception:
                known_classes = list(encoder.classes_)
                df[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in known_classes else -1)
        elif df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # 6️⃣ Escalar los datos (solo las features)
    try:
        X = scaler.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al escalar datos: {e}")

    # 7️⃣ Realizar predicción
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    # 8️⃣ (Opcional) Probabilidades si el modelo lo permite
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X).tolist()
        except Exception:
            probabilities = None

    # 9️⃣ Resultado
    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "n_samples": len(preds),
        "predictions": preds.tolist(),
        "probabilities": probabilities,
    }
