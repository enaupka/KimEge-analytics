
#Модуль детекции аномалий методом Тьюки (IQR)


import json
import math
import hashlib
import uuid
import numpy as np


def detect_anomalies_tukey(scores_matrix: np.ndarray, probabilities: np.ndarray, max_cards: int = 50) -> dict:
    """
    Выявление аномальных паттернов ответов (списывание / угадывание) по критерию Тьюки.
    Исключает автоматическое удаление данных (согласно ДП-05, аномалия — это сигнал для проверки).
    """
    n_check = min(10000, scores_matrix.shape[0])
    sub_scores = scores_matrix[:n_check]
    sub_probs = probabilities[:n_check]

    # Защита от деления на ноль при логарифмировании
    eps = 1e-15
    p = np.clip(sub_probs, eps, 1.0 - eps)
    
    # Расчет логарифмического правдоподобия (LL) бланка участника
    ll = np.sum(sub_scores * np.log(p) + (1.0 - sub_scores) * np.log(1.0 - p), axis=1)

    # Метод Тьюки (IQR) 
    q1, q3 = np.percentile(ll, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    
    anomaly_indices = np.where(ll < lower_fence)[0]

    cards = []
    now_iso = "2026-11-14T09:00:00.000Z"

    for idx in anomaly_indices[:max_cards]:
        pseudo_id = hashlib.sha256(f"salt_2026_student_{idx}".encode()).hexdigest()[:16]
        prim_score = int(sub_scores[idx].sum())
        
        cards.append({
            "anomalyId": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"work-{pseudo_id}")),
            "createdAt": now_iso,
            "type": "INVERTED_PROFILE",
            "objectKind": "room",
            "objectId": f"participant-{pseudo_id}",
            "metrics": [
                {"name": "log_likelihood", "value": round(float(ll[idx]), 4), "threshold": round(float(lower_fence), 4), "unit": "logit"},
                {"name": "primary_score", "value": prim_score, "threshold": 5, "unit": "балл"}
            ],
            "severity": "high" if ll[idx] < (q1 - 3.0 * iqr) else "medium",
            "recommendation": "Сигнал для проверки (ДП-05): аномальный профиль ответов по критерию Тьюки, автоматическое изменение баллов не производится",
            "dataWindow": {"from": "2026-06-08T07:00:00.000Z", "to": "2026-06-08T11:00:00.000Z"}
        })

    report_payload = {
        "schemaVersion": "1.0",
        "method": "Tukey IQR (Q1 - 1.5*IQR on Log-Likelihood)",
        "totalWorksChecked": int(n_check),
        "tukeyLowerFenceLL": round(float(lower_fence), 4),
        "anomaliesDetectedTotal": int(len(anomaly_indices)),
        "cardsExported": len(cards),
        "cards": cards
    }

    return report_payload