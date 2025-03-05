import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_aurc(outputs, targets):
    """
    AURC (Area Under the Risk-Coverage Curve) 계산 (정규화 포함)
    - outputs: 모델의 softmax 확률 출력 (numpy array)
    - targets: 정답 레이블 (numpy array)
    """
    confidences = np.max(outputs, axis=1)  # 각 샘플에서 최대 confidence
    predictions = np.argmax(outputs, axis=1)  # 예측된 클래스
    failures = (predictions != targets).astype(int)  # 실패 여부 (1: 실패, 0: 성공)

    # Confidence 기준으로 정렬
    sorted_indices = np.argsort(confidences)  # 신뢰도가 낮은 순으로 정렬
    sorted_failures = failures[sorted_indices]
    
    # Coverage 대비 Risk 누적 계산 (정규화)
    coverage = np.arange(1, len(failures) + 1) / len(failures)  # 0~1 범위 정규화
    risk = np.cumsum(sorted_failures) / np.arange(1, len(failures) + 1)  # 누적된 실패율 정규화

    # AURC 값 계산 (정규화된 범위로)
    aurc = np.trapz(risk, coverage)  # Risk-Coverage Curve의 면적 계산
    return aurc

def compute_auroc(outputs, targets):
    """
    AUROC (Area Under the Receiver Operating Characteristic Curve) 계산
    - outputs: 모델의 softmax 확률 출력 (numpy array)
    - targets: 정답 레이블 (numpy array)
    """
    confidences = np.max(outputs, axis=1)  # 최대 신뢰도
    predictions = np.argmax(outputs, axis=1)  # 예측값
    failures = (predictions != targets).astype(int)  # 실패 여부 (1: 실패, 0: 성공)

    # AUROC 계산
    auroc = roc_auc_score(failures, -confidences)  # 낮은 신뢰도일수록 Failure로 분류하도록 음수 변환
    return auroc

def compute_fpr95(outputs, targets):
    """
    FPR@95TPR (False Positive Rate at 95% True Positive Rate) 계산
    - outputs: 모델의 softmax 확률 출력 (numpy array)
    - targets: 정답 레이블 (numpy array)
    """
    confidences = np.max(outputs, axis=1)  # 최대 confidence
    predictions = np.argmax(outputs, axis=1)  # 예측값
    failures = (predictions != targets).astype(int)  # 실패 여부 (1: 실패, 0: 성공)

    # ROC Curve 계산
    fpr, tpr, thresholds = roc_curve(failures, -confidences)  # 음수로 변환하여 low confidence가 failure로 분류되도록 설정
    
    # 95% TPR이 되는 지점에서 FPR 값을 찾음
    idx_95tpr = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx_95tpr] if idx_95tpr < len(fpr) else 1.0  # 95TPR이 존재하지 않는 경우 FPR=1
    return fpr95
