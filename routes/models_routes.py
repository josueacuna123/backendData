from fastapi import APIRouter, HTTPException
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 📋 Obtener lista de modelos entrenados
@router.get("/models")
def get_models():
    try:
        response = supabase.table("models").select("*").order("trained_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener modelos: {str(e)}")

# 🗑️ Eliminar modelo (tabla + bucket)
@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    try:
        # Buscar modelo por id
        record = supabase.table("models").select("model_url").eq("id", model_id).maybe_single().execute()
        if not record.data:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")

        # Extraer nombre del archivo desde la URL
        model_url = record.data["model_url"]
        file_name = model_url.split("/")[-1]

        # Eliminar del bucket
        supabase.storage.from_("models").remove([file_name])

        # Eliminar de la tabla
        supabase.table("models").delete().eq("id", model_id).execute()

        return {"message": "Modelo eliminado correctamente"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar modelo: {str(e)}")
