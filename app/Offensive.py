"""
Offensive question detection - FastAPI server 
PLM 모델을 사용하여 질문이 offensive 한지 예측 
"""

import os 
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title= "Offensive - Offensive Question Detector")

class OffensiveDetectionRequest(BaseModel):
    """offensive detection request model"""
    question: str 

class OffensiveDetectionResponse(BaseModel):
    """offensive detection response model"""
    is_offensive: bool

class OffensiveDetectionModel:
    """offensive detection model"""
    def __init__(self):
        with open("configs/base_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.cache_dir = config['paths']["cache_dir"]
        
        # offensive 모델 경로 설정
        self.offensive_model_dir = config['counseling']["offensive_model_path"]
        self.model, self.tokenizer = self.load_offensive_model()
        
    def load_offensive_model(self):
        """offensive 모델 로드"""

        model = AutoModelForSequenceClassification.from_pretrained(
            self.offensive_model_dir,
            cache_dir=os.path.expanduser(self.cache_dir),
            torch_dtype=torch.float16,
        ).to(self.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.offensive_model_dir)
        
        return model, tokenizer
    
    def predict_offensive(self, question: str):
        """offensive 판별"""
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
            is_offensive = pred_id == 1
            return is_offensive



offensive_model = OffensiveDetectionModel()

@app.get("/")
async def root():
    """health check endpoint"""
    return {"status": "ok", "service": "Offensive - Offensive Question Detector"}

@app.post("/predict_offensive", response_model=OffensiveDetectionResponse)
async def predict_offensive(request: OffensiveDetectionRequest):
    """offensive detection endpoint"""
    try:
        is_offensive = offensive_model.predict_offensive(request.question)
        return {"is_offensive": is_offensive}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """health check endpoint"""
    return {
        "status": "healthy",
        "device": str(offensive_model.device),
        "loaded_model_path": offensive_model.offensive_model_dir
    }

if __name__ == "__main__":
    uvicorn.run(
        "Offensive:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )