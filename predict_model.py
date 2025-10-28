import os
import io
import pickle
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from supabase import create_client
from fastapi import HTTPException

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
# 🧠 Función de predicción individual
# ===============================
def predict_model(model_url: str, input_data):
    """
    Carga un modelo completo (modelo + scaler + encoders) desde Supabase Storage
    y realiza predicciones sobre nuevos datos.
    """

    # 🔹 Descargar el archivo del modelo
    response = requests.get(model_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Error al descargar el modelo: {response.text}")

    model_obj = pickle.loads(response.content)

    # 🔹 Extraer componentes
    model = model_obj.get("model")
    scaler = model_obj.get("scaler")
    encoders = model_obj.get("encoders", {})

    if model is None or scaler is None:
        raise HTTPException(status_code=400, detail="Modelo incompleto: faltan componentes 'model' o 'scaler'.")

    # 🔹 Convertir input a DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        raise HTTPException(status_code=400, detail="El parámetro 'input_data' debe ser un dict o lista de dicts.")

    # 🔹 Preprocesamiento igual al entrenamiento
    for col in df.columns:
        if col in encoders:
            encoder = encoders[col]
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except Exception:
                known_classes = list(encoder.classes_)
                df[col] = df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in known_classes else -1
                )
        elif df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # 🔹 Escalado
    try:
        X = scaler.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al escalar los datos: {e}")

    # 🔹 Predicción
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X).tolist()
        except Exception:
            probabilities = None

    # 🔹 Estructura uniforme para frontend
    return {
        "predictions": preds.tolist(),
        "probabilities": probabilities,
        "samples": len(preds)
    }


