from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pickle
import numpy as np
import json
import os
from pathlib import Path

app = FastAPI(
    title="Fabric Recommendation API",
    description="API untuk rekomendasi kain Pertenunan Astiti",
    version="1.0.0"
)

# CORS configuration untuk Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables untuk model dan data
model_package = None
fabric_data = []

# Pydantic models
class FabricQuery(BaseModel):
    warna_kain: Optional[str] = None
    jenis_kain: Optional[str] = None
    motif_kain: Optional[str] = None
    max_results: Optional[int] = 20

class FabricResponse(BaseModel):
    id: int
    warna_kain: str
    jenis_kain: str
    motif_kain: str
    foto_kain: Optional[str] = None
    deskripsi_kain: Optional[str] = None
    similarity_score: float
    matched_attributes: int
    total_attributes: int

class AIPredict(BaseModel):
    warna_kain: str
    jenis_kain: str
    motif_kain: str

class AIResponse(BaseModel):
    motif: str
    confidence: float

class RecommendationResult(BaseModel):
    recommendations: List[FabricResponse]
    total_found: int
    user_criteria: Dict[str, str]
    ai_predictions: Optional[List[AIResponse]] = None

@app.on_event("startup")
async def startup_event():
    await load_model()

async def load_model():
    global model_package, fabric_data

    model_path = Path("fabric_recommendation_model.pkl")

    try:
        if model_path.exists():
            with open(model_path, "rb") as f:
                model_package = pickle.load(f)
            print("✅ Model berhasil dimuat dari pickle file.")
        else:
            print("⚠️ File model tidak ditemukan. Menggunakan model dummy untuk fallback.")
            model_package = create_dummy_model()

        fabric_data = model_package.get("fabric_data", [])
        print(f"📦 Model keys: {list(model_package.keys())}")
        print(f"🧵 Jumlah data kain: {len(fabric_data)}")

    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        print("⚠️ Menggunakan model dummy untuk fallback.")
        model_package = create_dummy_model()
        fabric_data = model_package.get("fabric_data", [])

def create_dummy_model():
    return {
        'fabric_data': [
            {
                'id': 0,
                'warna_kain': 'Merah',
                'jenis_kain': 'Sutra',
                'motif_kain': 'Parang',
                'foto_kain': 'sample1.jpg',
                'deskripsi_kain': 'Kain sutra merah dengan motif parang'
            },
            {
                'id': 1,
                'warna_kain': 'Biru',
                'jenis_kain': 'Katun',
                'motif_kain': 'Zigzag',
                'foto_kain': 'sample2.jpg',
                'deskripsi_kain': 'Kain katun biru dengan motif geometris'
            }
        ],
        'unique_values': {
            'warna_kain': ['Merah', 'Biru', 'Hijau'],
            'jenis_kain': ['Sutra', 'Katun', 'Linen'],
            'motif_kain': ['Parang', 'Geometris', 'Bunga']
        }
    }

def calculate_similarity_score(user_input: dict, fabric_data: dict) -> tuple:
    score = 0
    total_attributes = 0

    for key in ['warna_kain', 'jenis_kain', 'motif_kain']:
        if user_input.get(key):
            total_attributes += 1
            if fabric_data[key].lower() == user_input[key].lower():
                score += 1

    similarity_percentage = (score / total_attributes * 100) if total_attributes > 0 else 0
    return similarity_percentage, score, total_attributes

def get_ai_predictions(warna_kain: str, jenis_kain: str, motif_kain: str) -> List[AIResponse]:
    try:
        if not model_package or 'model' not in model_package:
            return []

        warna_encoder = model_package['warna_encoder']
        jenis_encoder = model_package['jenis_encoder']
        motif_encoder = model_package['motif_encoder']
        model = model_package['model']

        try:
            warna_encoded = warna_encoder.transform([warna_kain])[0]
            jenis_encoded = jenis_encoder.transform([jenis_kain])[0]
            motif_encoded = motif_encoder.transform([motif_kain])[0]
        except Exception as e:
            print("Encoding error:", e)
            raise HTTPException(status_code=400, detail="Encoding gagal, pastikan input valid")

        input_data = [[warna_encoded, jenis_encoded, motif_encoded]]

        try:
            prediction_proba = model.predict_proba(input_data)[0]
            top_indices = np.argsort(prediction_proba)[-3:][::-1]

            predictions = []
            for idx in top_indices:
                motif = motif_encoder.inverse_transform([idx])[0]
                confidence = float(prediction_proba[idx] * 100)
                predictions.append(AIResponse(motif=motif, confidence=confidence))

            return predictions
        except:
            predicted = model.predict(input_data)[0]
            motif = motif_encoder.inverse_transform([predicted])[0]
            return [AIResponse(motif=motif, confidence=85.0)]

    except Exception as e:
        print(f"Error in AI prediction: {e}")
        return []

@app.get("/")
async def root():
    return {
        "message": "Fabric Recommendation API",
        "version": "1.0.0",
        "status": "active",
        "total_fabrics": len(fabric_data)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_package is not None,
        "fabric_count": len(fabric_data)
    }

@app.get("/fabric-options")
async def get_fabric_options():
    if not model_package:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "warna_kain": model_package.get('unique_values', {}).get('warna_kain', []),
        "jenis_kain": model_package.get('unique_values', {}).get('jenis_kain', []),
        "motif_kain": model_package.get('unique_values', {}).get('motif_kain', [])
    }

@app.post("/recommend", response_model=RecommendationResult)
async def get_recommendations(query: FabricQuery):
    if not fabric_data:
        raise HTTPException(status_code=500, detail="Fabric data not loaded")

    user_input = {}
    if query.warna_kain:
        user_input['warna_kain'] = query.warna_kain.strip()
    if query.jenis_kain:
        user_input['jenis_kain'] = query.jenis_kain.strip()
    if query.motif_kain:
        user_input['motif_kain'] = query.motif_kain.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="At least one search criteria is required")

    recommendations = []
    for fabric in fabric_data:
        similarity_score, matched_attrs, total_attrs = calculate_similarity_score(user_input, fabric)

        if similarity_score > 0:
            recommendations.append(FabricResponse(
                id=fabric['id'],
                warna_kain=fabric['warna_kain'],
                jenis_kain=fabric['jenis_kain'],
                motif_kain=fabric['motif_kain'],
                foto_kain=fabric.get('foto_kain'),
                deskripsi_kain=fabric.get('deskripsi_kain'),
                similarity_score=similarity_score,
                matched_attributes=matched_attrs,
                total_attributes=total_attrs
            ))

    recommendations.sort(key=lambda x: x.similarity_score, reverse=True)

    max_results = min(query.max_results or 20, 50)
    recommendations = recommendations[:max_results]

    ai_predictions = []
    if query.warna_kain and query.jenis_kain and query.motif_kain:
        ai_predictions = get_ai_predictions(query.warna_kain, query.jenis_kain, query.motif_kain)

    return RecommendationResult(
        recommendations=recommendations,
        total_found=len(recommendations),
        user_criteria=user_input,
        ai_predictions=ai_predictions if ai_predictions else None
    )

@app.post("/predict", response_model=List[AIResponse])
async def predict_motif(query: AIPredict):
    if not model_package:
        raise HTTPException(status_code=500, detail="Model not loaded")

    predictions = get_ai_predictions(query.warna_kain, query.jenis_kain, query.motif_kain)

    if not predictions:
        raise HTTPException(status_code=400, detail="Unable to generate predictions")

    return predictions

@app.get("/fabric/{fabric_id}")
async def get_fabric_by_id(fabric_id: int):
    if not fabric_data:
        raise HTTPException(status_code=500, detail="Fabric data not loaded")

    fabric = next((f for f in fabric_data if f['id'] == fabric_id), None)

    if not fabric:
        raise HTTPException(status_code=404, detail="Fabric not found")

    return fabric

@app.get("/stats")
async def get_statistics():
    if not fabric_data:
        raise HTTPException(status_code=500, detail="Fabric data not loaded")

    warna_dist = {}
    jenis_dist = {}
    motif_dist = {}

    for fabric in fabric_data:
        warna = fabric['warna_kain']
        jenis = fabric['jenis_kain']
        motif = fabric['motif_kain']

        warna_dist[warna] = warna_dist.get(warna, 0) + 1
        jenis_dist[jenis] = jenis_dist.get(jenis, 0) + 1
        motif_dist[motif] = motif_dist.get(motif, 0) + 1

    return {
        "total_fabrics": len(fabric_data),
        "unique_colors": len(warna_dist),
        "unique_types": len(jenis_dist),
        "unique_motifs": len(motif_dist),
        "color_distribution": dict(sorted(warna_dist.items(), key=lambda x: x[1], reverse=True)),
        "type_distribution": dict(sorted(jenis_dist.items(), key=lambda x: x[1], reverse=True)),
        "motif_distribution": dict(sorted(motif_dist.items(), key=lambda x: x[1], reverse=True)),
        "model_info": model_package.get('model_info', {}) if model_package else {}
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
