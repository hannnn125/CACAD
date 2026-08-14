"""
NQCP (Next Question Cluster Predictor) - FastAPI 서버
PLM 모델을 사용하여 다음 질문 클러스터를 예측하는 서비스
"""
import os
import json
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Dict
import uvicorn


app = FastAPI(title="NQCP - Next Question Cluster Predictor")


class ClusterPredictionRequest(BaseModel):
    """클러스터 예측 요청 모델"""
    history_text: str
    abuse_type: str


class ClusterProb(BaseModel):
    """클러스터 확률 모델"""
    cluster_id: int
    probability: float


class ClusterPredictionResponse(BaseModel):
    """클러스터 예측 응답 모델"""
    cluster_probs: List[ClusterProb]


class NQCPModel:
    """클러스터 예측 모델 관리 클래스"""
    
    def __init__(self):
        with open("configs/base_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.cluster_models = {}
        self.cache_dir = config['paths']["cache_dir"]
        
        # 클러스터 모델 경로 설정
        self.cls_model_paths = config['counseling']["nqcp_model_paths"]
    
    def _load_cluster_model(self, abuse_type: str):
        """클러스터 예측 모델 로드 (캐시 지원)"""
        if abuse_type in self.cluster_models:
            return self.cluster_models[abuse_type]
        
        print(f"클러스터 모델 로딩 중: {abuse_type}")
        
        if abuse_type not in self.cls_model_paths:
            raise ValueError(f"클러스터 모델 경로가 설정되지 않았습니다: {abuse_type}")
        
        cls_model_dir = self.cls_model_paths[abuse_type]
        print(f"  → 모델 경로: {cls_model_dir}")
        
        cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_dir)
        cls_model = AutoModelForSequenceClassification.from_pretrained(
            cls_model_dir,
            cache_dir=os.path.expanduser(self.cache_dir),
            torch_dtype=torch.float16
        ).to(self.device)
        
        # label2id.json 파일에서 실제 클러스터 ID 매핑 로드
        label2id_path = os.path.join(cls_model_dir, "label2id.json")
        id2cluster = {}
        
        if os.path.exists(label2id_path):
            print(f"  → label2id.json 파일 발견: {label2id_path}")
            with open(label2id_path, 'r', encoding='utf-8') as f:
                label2id = json.load(f)
            # label2id를 id2cluster로 변환 (역매핑)
            # label2id: {"0": 0, "1": 1, ...} -> id2cluster: {0: 0, 1: 1, ...}
            id2cluster = {v: int(k) for k, v in label2id.items()}
            print(f"  → label2id.json에서 로드: {len(id2cluster)}개 클러스터")
        else:
            # label2id.json이 없으면 config.id2label 사용
            print(f"  → label2id.json 파일 없음, config.id2label 사용")
            id2label = cls_model.config.id2label
            # id2label이 LABEL_0 형식이면 숫자 추출
            if isinstance(id2label, dict):
                for idx, label in id2label.items():
                    if isinstance(label, str) and label.startswith('LABEL_'):
                        try:
                            cluster_id = int(label.replace('LABEL_', ''))
                            id2cluster[int(idx)] = cluster_id
                        except ValueError:
                            id2cluster[int(idx)] = int(idx)
                    else:
                        try:
                            id2cluster[int(idx)] = int(label)
                        except (ValueError, TypeError):
                            id2cluster[int(idx)] = int(idx)
            else:
                # 리스트인 경우
                for idx, label in enumerate(id2label):
                    if isinstance(label, str) and label.startswith('LABEL_'):
                        try:
                            cluster_id = int(label.replace('LABEL_', ''))
                            id2cluster[idx] = cluster_id
                        except ValueError:
                            id2cluster[idx] = idx
                    else:
                        try:
                            id2cluster[idx] = int(label)
                        except (ValueError, TypeError):
                            id2cluster[idx] = idx
        
        # 디버깅: id2cluster 내용 출력
        print(f"  → id2cluster 타입: {type(id2cluster)}")
        if isinstance(id2cluster, dict):
            print(f"  → id2cluster 샘플 (처음 5개): {dict(list(id2cluster.items())[:5])}")
        
        # 캐시에 저장
        self.cluster_models[abuse_type] = (cls_tokenizer, cls_model, id2cluster)
        print(f"클러스터 모델 로딩 완료: {abuse_type}")
        
        return cls_tokenizer, cls_model, id2cluster
    
    def predict_clusters_with_probs(self, history_text: str, abuse_type: str) -> List[Tuple[int, float]]:
        """클러스터 예측 결과와 확률값 반환"""
        cls_tokenizer, cls_model, id2cluster = self._load_cluster_model(abuse_type)
        
        cls_inputs = cls_tokenizer(
            history_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = cls_model(**cls_inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        # id2cluster 딕셔너리에서 클러스터 ID 가져오기
        # id2cluster는 {모델_인덱스: 실제_클러스터_ID} 형식
        cluster_probs = []
        num_labels = len(probs)
        
        for i in range(num_labels):
            # id2cluster에서 실제 클러스터 ID 가져오기
            if isinstance(id2cluster, dict):
                cluster_id = id2cluster.get(i, i)  # 없으면 인덱스 자체 사용
            else:
                cluster_id = i  # 기본값으로 인덱스 사용
            
            cluster_probs.append((int(cluster_id), float(probs[i])))
        
        cluster_probs.sort(key=lambda x: x[1], reverse=True)
        
        return cluster_probs


# 전역 모델 인스턴스
nqcp_model = NQCPModel()


@app.get("/")
async def root():
    """헬스 체크 엔드포인트"""
    return {"status": "ok", "service": "NQCP - Next Question Cluster Predictor"}


@app.post("/predict_cluster", response_model=ClusterPredictionResponse)
async def predict_cluster(request: ClusterPredictionRequest):
    """클러스터 예측 엔드포인트"""
    try:
        # 요청 데이터 검증
        if not request.abuse_type:
            raise HTTPException(
                status_code=400, 
                detail=f"abuse_type이 필요합니다. 받은 값: {request.abuse_type}"
            )
        
        if request.abuse_type not in nqcp_model.cls_model_paths:
            available_types = list(nqcp_model.cls_model_paths.keys())
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 abuse_type입니다: '{request.abuse_type}'. 지원되는 타입: {available_types}"
            )
        
        if request.history_text is None:
            request.history_text = ""
        
        print(f"[NQCP] 요청 받음 - abuse_type: {request.abuse_type}, history_text 길이: {len(request.history_text)}")
        
        cluster_probs = nqcp_model.predict_clusters_with_probs(
            request.history_text,
            request.abuse_type
        )
        
        # 응답 형식으로 변환
        response = ClusterPredictionResponse(
            cluster_probs=[
                ClusterProb(cluster_id=cluster_id, probability=prob)
                for cluster_id, prob in cluster_probs
            ]
        )
        
        print(f"[NQCP] 예측 완료 - {len(cluster_probs)}개 클러스터 반환")
        return response
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"값 오류: {str(e)}")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[NQCP] 예측 중 오류 발생:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")


@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    return {
        "status": "healthy",
        "device": str(nqcp_model.device),
        "loaded_models": list(nqcp_model.cluster_models.keys())
    }


if __name__ == "__main__":
    uvicorn.run(
        "NQCP:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )
