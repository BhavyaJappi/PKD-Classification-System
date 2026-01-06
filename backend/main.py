"""
PKD Classification Backend - FastAPI Server
============================================

Lightweight FastAPI server for kidney CT image classification.
Serves the trained EfficientNet-B0 model for inference.
"""

import io
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torchvision.transforms as T
from torchvision import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class Settings:
    """Application settings"""
    MODEL_PATH = Path("../outputs/models/best_model_fold_1.pth")
    CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
    IMAGE_SIZE = 224
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_BATCH_SIZE = 10

settings = Settings()

# ============================================================================
# Model Definition (must match training)
# ============================================================================

class KidneyClassifier(nn.Module):
    """EfficientNet-B0 based classifier for kidney CT images"""
    
    def __init__(self, num_classes=4, pretrained=False, dropout_rate=0.3):
        super(KidneyClassifier, self).__init__()
        
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

# ============================================================================
# Model Loading
# ============================================================================

def load_model() -> Optional[KidneyClassifier]:
    """Load the trained model"""
    model_path = settings.MODEL_PATH
    
    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}")
        return None
    
    try:
        model = KidneyClassifier(num_classes=4, pretrained=False)
        checkpoint = torch.load(model_path, map_location=settings.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(settings.DEVICE)
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# ============================================================================
# Image Preprocessing
# ============================================================================

def get_transforms():
    """Get preprocessing transforms"""
    return T.Compose([
        T.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image bytes for inference"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = get_transforms()
    tensor = transform(image).unsqueeze(0)
    return tensor

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="PKD Classification API",
    description="API for classifying kidney CT images (Normal, Cyst, Tumor, Stone)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[KidneyClassifier] = None

# ============================================================================
# Response Models
# ============================================================================

class PredictionResult(BaseModel):
    """Single prediction result"""
    class_name: str
    confidence: float
    all_probabilities: dict

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str

class BatchPredictionResult(BaseModel):
    """Batch prediction result"""
    predictions: List[PredictionResult]
    total_images: int

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    model = load_model()
    if model is None:
        logger.warning("Model not loaded. Predictions will be unavailable.")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "PKD Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=settings.DEVICE
    )

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predict kidney condition from a single CT image.
    
    Upload a kidney CT scan image (JPG, PNG, or JPEG) to get classification.
    Returns the predicted class and confidence scores.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes).to(settings.DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
        
        # Prepare response
        all_probs = {
            name: float(prob)
            for name, prob in zip(settings.CLASS_NAMES, probabilities.cpu().numpy())
        }
        
        return PredictionResult(
            class_name=settings.CLASS_NAMES[predicted.item()],
            confidence=float(confidence.item()),
            all_probabilities=all_probs
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict kidney conditions from multiple CT images.
    
    Upload up to 10 kidney CT scan images to get batch classification.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.MAX_BATCH_SIZE} images allowed per batch"
        )
    
    predictions = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        
        try:
            image_bytes = await file.read()
            tensor = preprocess_image(image_bytes).to(settings.DEVICE)
            
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
            
            all_probs = {
                name: float(prob)
                for name, prob in zip(settings.CLASS_NAMES, probabilities.cpu().numpy())
            }
            
            predictions.append(PredictionResult(
                class_name=settings.CLASS_NAMES[predicted.item()],
                confidence=float(confidence.item()),
                all_probabilities=all_probs
            ))
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            continue
    
    return BatchPredictionResult(
        predictions=predictions,
        total_images=len(predictions)
    )

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    return {
        "model_name": "EfficientNet-B0",
        "num_classes": len(settings.CLASS_NAMES),
        "class_names": settings.CLASS_NAMES,
        "image_size": settings.IMAGE_SIZE,
        "device": settings.DEVICE,
        "model_loaded": model is not None
    }

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
