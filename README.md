# MLOps Workflow Test Repository

Simple Python functions for testing MLOps workflow system.

## 파일 구조

```
mlops-test-repo/
├── src/
│   ├── data.py       # 데이터 로딩 및 전처리 함수
│   └── train.py      # 모델 학습, 평가, 배포 함수
├── requirements.txt  # 의존성 (없음)
└── README.md
```

## 사용 가능한 함수

### src/data.py
- `load_data()` - 샘플 데이터 로딩
- `preprocess_data()` - 데이터 전처리

### src/train.py
- `train_model()` - 모델 학습
- `evaluate_model()` - 모델 평가
- `deploy_model()` - 모델 배포

## 테스트 방법

```python
# 로컬 테스트
from src.data import load_data, preprocess_data
from src.train import train_model, evaluate_model, deploy_model

# 데이터 파이프라인
load_data()
preprocess_data()

# 모델 파이프라인
train_model()
evaluate_model()
deploy_model()
```

## MLOps Workflow 설정 예시

**Workflow 1: Data Pipeline**
1. Task: `load_data` → `src/data.py` → `load_data()`
2. Task: `preprocess_data` → `src/data.py` → `preprocess_data()`

**Workflow 2: Model Training Pipeline**
1. Task: `train_model` → `src/train.py` → `train_model()`
2. Task: `evaluate_model` → `src/train.py` → `evaluate_model()`
3. Task: `deploy_model` → `src/train.py` → `deploy_model()`
