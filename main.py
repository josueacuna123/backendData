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
from backend.predict_all import predict_all



# ===============================
# 🚀 Configurar FastAPI
# ===============================
app = FastAPI(title="DataFlow ML API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(models_routes.router)


# ===============================
# 🌍 Variables de entorno
# ===============================
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Conexión exitosa con Supabase")

# ===============================
# 📦 Request model
# ===============================
class TrainRequest(BaseModel):
    dataset_id: str
    model_type: str
    target_column: str

# ===============================
# 🌐 Rutas base
# ===============================
@app.get("/")
def root():
    return {"message": "🚀 Backend operativo y listo para entrenar modelos"}

# ===============================
# 🧠 Entrenamiento REST
# ===============================
@app.post("/train")
def train(req: TrainRequest):
    try:
        result = train_model(req.dataset_id, req.model_type, req.target_column)
        return {"status": "ok", "metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# 🔴 Entrenamiento vía WebSocket
# ===============================
@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    try:
        config = await websocket.receive_json()
        dataset_id = config.get("dataset_id")
        model_type = config.get("model_type")
        target_column = config.get("target_column")

        await websocket.send_text("🔌 Conectando al backend...")
        await asyncio.sleep(0.2)
        await websocket.send_text("✅ Conexión establecida. Iniciando entrenamiento...")
        await asyncio.sleep(0.5)
        await websocket.send_text(f"📦 Dataset: {dataset_id}")
        await websocket.send_text(f"🧠 Modelo: {model_type}")
        await websocket.send_text(f"🎯 Target: {target_column}")

        await asyncio.to_thread(train_model, dataset_id, model_type, target_column, websocket)
        await websocket.send_text("✅ Entrenamiento completado correctamente.")
        await websocket.close()

    except WebSocketDisconnect:
        print("⚠️ Cliente desconectado durante el entrenamiento.")
    except Exception as e:
        await websocket.send_text(f"❌ Error: {str(e)}")
        await websocket.close()
        
class PredictRequest(BaseModel):
    model_url: str
    input_data: dict
    expected_features: list = []  # opcional, si tu frontend lo envía


router = APIRouter()

@router.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        # ✅ Ejecutar predicción (mantiene tu lógica original)
        result = predict_model(req.model_url, req.input_data)

        # ✅ Buscar el ID del modelo asociado en la tabla "models"
        model_record = (
            supabase.table("models")
            .select("id")
            .eq("model_url", req.model_url)
            .maybe_single()
            .execute()
        )

        model_id = None
        if model_record.data and "id" in model_record.data:
            model_id = model_record.data["id"]

        # ✅ Insertar la predicción individual en la tabla "predictions"
        supabase.table("predictions").insert({
            "model_id": model_id,
            "input_data": req.input_data,
            "output_data": result,  # Guarda todo el JSON devuelto
        }).execute()

        # ✅ Retornar el resultado como antes (sin alterar nada)
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/models")
def get_models():
    """
    Retorna la lista de modelos disponibles desde Supabase.
    """
    try:
        response = supabase.table("models").select("*").execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="No se encontraron modelos almacenados")

        # 🔹 Convertimos los modelos a un formato legible por el frontend
        models_list = [
            {
                "id": row.get("id"),
                "model_name": row.get("model_name"),
                "algorithm": row.get("algorithm"),
                "task_type": row.get("task_type"),
                "accuracy": row.get("accuracy"),
                "target_column": row.get("target"),
                "features": row.get("features", []),
                "model_url": row.get("model_url"),
                "trained_at": row.get("trained_at"),
                "dataset_id": row.get("dataset_id"),
            }
            for row in response.data
        ]

        return models_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener modelos: {e}")
    
    
@app.get("/models/{dataset_id}")
def list_models_by_dataset(dataset_id: str):
    try:
        response = supabase.table("models").select("*").eq("dataset_id", dataset_id).execute()
        return {"status": "ok", "models": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/predict/all")
def predict_all_endpoint(req: PredictRequest):
    try:
        # Obtener dataset_id (puede venir desde input_data)
        dataset_id = req.input_data.get("dataset_id") if isinstance(req.input_data, dict) else None
        if not dataset_id:
            raise HTTPException(status_code=400, detail="Falta el dataset_id para la predicción masiva.")

        result = predict_all(req.model_url, dataset_id)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



app.include_router(router)  # 👈 AÑADE ESTA LÍNEA