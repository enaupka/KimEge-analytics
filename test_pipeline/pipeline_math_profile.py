# -*- coding: utf-8 -*-

#pipeline

import json
import math
import hashlib
import uuid
import numpy as np
import pandas as pd


TASK_COMPLETION_RATE_AVG = (
    85.67, 90.53, 70.27, 94.07, 71.80, 95.70,
    75.37, 71.40, 77.53, 75.90, 81.73, 75.27
)
TASK_COMPLETION_RATE_2026 = (
    92.4, 92.5, 73.2, 94.5, 77.6, 95.5, 82.7, 77.4, 80.7, 78.4, 86.1, 76.7
)
TASK_COMPLETION_RATE_2025 = (
    82.4, 84.9, 69.9, 92.8, 72.6, 97.1, 59.0, 61.2, 68.5, 75.5, 84.7, 69.6
)
TASK_COMPLETION_RATE_2024 = (
    82.2, 94.2, 67.7, 94.9, 65.2, 94.5, 84.4, 75.6, 83.4, 73.8, 74.4, 79.5
)

CODEFICATOR_MAP = [
    "5.1.1", "7.1.1", "8.1.1", "8.2.1", "2.1.1", "1.1.1",
    "4.1.1", "7.2.1", "3.1.1", "3.2.1", "4.2.1", "4.3.1"
]

s = 0.99
k = 1.235


def calculate_difficulty(a: float) -> float:
    return -math.log(a / (1.0 - a))


def simulate_exam_session(task_completion_rate, n_students=500000, k_val=k, s_val=s, seed=42):
    """Генерация синтетической выборки ответов участников (NFR-01, NFR-08)."""
    rng = np.random.default_rng(seed)
    p_targets = np.asarray(task_completion_rate, dtype=float) / 100.0
    beta = np.array([k_val * calculate_difficulty(x) for x in p_targets]).reshape(1, -1)
    theta = rng.normal(loc=0.0, scale=1.0, size=(n_students, 1))
    probabilities = s_val / (1.0 + np.exp(-(theta - beta)))
    scores = (rng.random((n_students, len(p_targets))) < probabilities).astype(int)
    return scores, probabilities


def build_primary_to_secondary_scale(
    task_completion_rate,
    k_val=k,
    s_val=s,
    secondary_max=70,
    n_groups=4001
):
    """Построение шкалы перевода первичных баллов (0..12) в тестовые (0..70) через EAP (FR-01, FR-03)."""
    task_completion_percentage = tuple(x / 100.0 for x in task_completion_rate)
    task_difficulty = tuple(k_val * calculate_difficulty(x) for x in task_completion_percentage)
    beta = np.array(task_difficulty).reshape(1, -1)

    theta_grid = np.linspace(-6.0, 6.0, n_groups).reshape(-1, 1)
    weights = np.exp(-0.5 * (theta_grid ** 2)) / np.sqrt(2.0 * np.pi)
    weights = weights / weights.sum()

    task_probs = s_val / (1.0 + np.exp(-(theta_grid - beta)))

    n_tasks = len(task_completion_rate)
    score_probs = np.zeros((n_groups, n_tasks + 1))
    score_probs[:, 0] = 1.0

    for i in range(n_tasks):
        p_i = task_probs[:, i:i + 1]
        next_score_probs = np.zeros_like(score_probs)
        next_score_probs[:, :i + 1] = score_probs[:, :i + 1] * (1.0 - p_i)
        next_score_probs[:, 1:i + 2] += score_probs[:, :i + 1] * p_i
        score_probs = next_score_probs

    joint_probs = score_probs * weights
    prob_of_score = joint_probs.sum(axis=0)
    theta_expected = (joint_probs * theta_grid).sum(axis=0) / prob_of_score

    theta_min = theta_expected[0]
    theta_max = theta_expected[-1]

    secondary_exact = secondary_max * (theta_expected - theta_min) / (theta_max - theta_min)
    secondary_integer = np.round(secondary_exact).astype(int)

    return pd.DataFrame({
        "Первичный балл": np.arange(n_tasks + 1),
        "Ожидаемый уровень (theta)": theta_expected,
        "Вторичный балл (точный)": secondary_exact,
        "Вторичный балл": secondary_integer,
        "Доля участников, %": prob_of_score * 100.0,
    })


def validate_data_and_scale(scores_matrix, scale_df):
    """Автоматический контроль качества данных и свойств шкалы (FR-04, NFR-02, NFR-03)."""
    assert scores_matrix.ndim == 2 and scores_matrix.shape[1] == 12, "Ожидается ровно 12 заданий"
    assert not np.isnan(scores_matrix).any(), "Обнаружены пропуски (NaN)"
    assert np.isin(scores_matrix, [0, 1]).all(), "Ответы должны быть строго бинарными (0 или 1)"

    prim = scale_df["Первичный балл"].to_numpy()
    sec = scale_df["Вторичный балл"].to_numpy()
    theta_eap = scale_df["Ожидаемый уровень (theta)"].to_numpy()

    assert np.array_equal(prim, np.arange(13)), "Шкала должна покрывать все баллы 0..12"
    assert np.all(np.diff(theta_eap) > 0), "Оценки способности EAP должны строго возрастать"
    assert np.all(np.diff(sec) >= 0), "Нарушена монотонность тестовых баллов"
    assert sec[0] == 0 and sec[-1] == 70, "Нарушены граничные точки (0 -> 0, 12 -> 70)"
    return True


def calculate_psychometrics_tz(scores_matrix, rates_2025=TASK_COMPLETION_RATE_2025):
    """Расчёт Альфы Кронбаха, решаемости p и индекса дифференциации D по крайним 27% (FR-06, FR-07, FR-09)."""
    scores = np.asarray(scores_matrix, dtype=float)
    n_students, n_tasks = scores.shape
    total_scores = scores.sum(axis=1)

    var_items = scores.var(axis=0, ddof=1)
    var_total = total_scores.var(ddof=1)
    cronbach_alpha = float((n_tasks / (n_tasks - 1)) * (1.0 - var_items.sum() / var_total))

    k_27 = int(np.round(n_students * 0.27))
    sorted_idx = np.argsort(total_scores)
    bottom_27_idx = sorted_idx[:k_27]
    top_27_idx = sorted_idx[-k_27:]

    p_all = scores.mean(axis=0)
    p_top = scores[top_27_idx].mean(axis=0)
    p_bottom = scores[bottom_27_idx].mean(axis=0)
    d_values = p_top - p_bottom
    correct_counts = scores.sum(axis=0).astype(int)

    flags_ru = []
    tasks_json = []
    for i in range(n_tasks):
        d_val = float(d_values[i])
        p_val = float(p_all[i])
        p_2025 = rates_2025[i] / 100.0

        if d_val < 0:
            flags_ru.append("отрицательная дифференциация")
            flag_code = "NEGATIVE_DISCRIMINATION"
        elif d_val < 0.2:
            flags_ru.append("низкая дифференциация")
            flag_code = "LOW_DISCRIMINATION"
        elif abs(p_val - p_2025) > 0.10:
            flags_ru.append("отклонение от ожидания")
            flag_code = "DEVIATION_FROM_2025"
        else:
            flags_ru.append("норма")
            flag_code = None

        tasks_json.append({
            "taskId": f"task-{i+1}",
            "codeficator": CODEFICATOR_MAP[i],
            "answered": int(n_students),
            "correct": int(correct_counts[i]),
            "pValue": round(p_val, 3),
            "avgScore": round(p_val, 3),
            "skipRate": 0.0,
            "discrimination": round(d_val, 3),
            "flag": flag_code
        })

    report_df = pd.DataFrame({
        "Задание": [f"№{i+1}" for i in range(n_tasks)],
        "Кодификатор": CODEFICATOR_MAP,
        "p (решаемость)": np.round(p_all, 3),
        "p (верхние 27%)": np.round(p_top, 3),
        "p (нижние 27%)": np.round(p_bottom, 3),
        "D (индекс дифф.)": np.round(d_values, 3),
        "Флаг": flags_ru,
    })

    flagged_tasks_df = report_df[report_df["Флаг"] != "норма"].copy()
    return cronbach_alpha, report_df, flagged_tasks_df, tasks_json


def detect_anomalies_tukey(scores_matrix, probabilities, scale_df, max_cards=50):
    """Детекция аномалий методом Тьюки (IQR) по логарифмическому правдоподобию и средним баллам ППЭ (FR-11, FR-13, FR-14)."""
    n_check = min(10000, scores_matrix.shape[0])
    sub_scores = scores_matrix[:n_check]
    sub_probs = probabilities[:n_check]

    eps = 1e-15
    p = np.clip(sub_probs, eps, 1.0 - eps)
    ll = np.sum(sub_scores * np.log(p) + (1.0 - sub_scores) * np.log(1.0 - p), axis=1)

    q1, q3 = np.percentile(ll, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    anomaly_indices = np.where(ll < lower_fence)[0]

    lookup = scale_df["Вторичный балл"].to_numpy()
    test_scores = lookup[sub_scores.sum(axis=1)]
    n_ppe = 100
    ppe_scores = test_scores.reshape(n_ppe, -1).mean(axis=1)
    ppe_q1, ppe_q3 = np.percentile(ppe_scores, [25, 75])
    ppe_upper_fence = ppe_q3 + 1.5 * (ppe_q3 - ppe_q1)

    cards = []
    now_iso = "2026-11-14T09:00:00.000Z"

    for ppe_idx in np.where(ppe_scores > ppe_upper_fence)[0]:
        cards.append({
            "anomalyId": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"ppe-{ppe_idx}")),
            "createdAt": now_iso,
            "type": "DISTRICT_OUTLIER",
            "objectKind": "ppe",
            "objectId": f"ppe-{1000 + int(ppe_idx)}",
            "metrics": [{
                "name": "avg_test_score",
                "value": round(float(ppe_scores[ppe_idx]), 2),
                "threshold": round(float(ppe_upper_fence), 2),
                "unit": "балл"
            }],
            "severity": "medium",
            "recommendation": "Передать в Ситуационный центр для проверки",
            "dataWindow": {"from": "2026-06-08T07:00:00.000Z", "to": "2026-06-08T11:00:00.000Z"}
        })

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

    return {
        "schemaVersion": "1.0",
        "method": "Tukey IQR (Q1 - 1.5*IQR on Log-Likelihood & Q3 + 1.5*IQR on PPE avg)",
        "totalWorksChecked": int(n_check),
        "tukeyLowerFenceLL": round(float(lower_fence), 4),
        "anomaliesDetectedTotal": int(len(anomaly_indices)),
        "cardsExported": len(cards),
        "cards": cards
    }


def export_all_json_contracts(scores_matrix, probs_matrix, scale_df):
    """Формирование и сохранение всех JSON-артефактов по разделу 7 ТЗ."""
    # 1. Шкала перевода (п. 7.2 ТЗ)
    scale_json = {
        "schemaVersion": "1.0",
        "subject": "math",
        "level": "profile",
        "year": 2026,
        "maxPrimary": 12,
        "minThresholdPrimary": 5,
        "entries": [
            {"primary": int(row["Первичный балл"]), "test": int(row["Вторичный балл"])}
            for _, row in scale_df.iterrows()
        ],
        "note": "Метод: модифицированная модель Раша (1PL IRT с верхней асимптотой s=0.99, k=1.235) + EAP (методика v1.0)"
    }
    with open("scale_math_profile_2026.json", "w", encoding="utf-8") as f:
        json.dump(scale_json, f, ensure_ascii=False, indent=2)

    # 2. Статистика заданий (п. 7.3 ТЗ)
    alpha, report_df, flagged_df, tasks_json = calculate_psychometrics_tz(scores_matrix)
    task_stats_json = {
        "sessionId": "2026-06-08-math-prof-01",
        "subject": "math",
        "level": "profile",
        "year": 2026,
        "cronbachAlpha": round(alpha, 4),
        "tasks": tasks_json,
        "distractors": []
    }
    with open("task-stats.json", "w", encoding="utf-8") as f:
        json.dump(task_stats_json, f, ensure_ascii=False, indent=2)

    # 3. Карточки аномалий (п. 7.4 ТЗ)
    anomalies_json = detect_anomalies_tukey(scores_matrix, probs_matrix, scale_df)
    with open("anomalies_cards_2026.json", "w", encoding="utf-8") as f:
        json.dump(anomalies_json, f, ensure_ascii=False, indent=2)

    # 4. Эталонный набор >= 500 пар с контрольной суммой SHA-256 (FR-04, NFR-02)
    lookup = scale_df["Вторичный балл"].to_numpy()
    cases = []
    for idx in range(500):
        pseudo = hashlib.sha256(f"ref_2026_{idx}".encode()).hexdigest()[:16]
        prim = int(scores_matrix[idx].sum())
        sec = int(lookup[prim])
        cases.append({
            "caseId": idx + 1,
            "participantPseudonym": pseudo,
            "responses": scores_matrix[idx].tolist(),
            "primaryScore": prim,
            "expectedTestScore": sec,
            "passedMinThreshold": bool(prim >= 5)
        })
    checksum = hashlib.sha256(json.dumps(cases, sort_keys=True).encode("utf-8")).hexdigest()
    ref_json = {
        "schemaVersion": "1.0",
        "subject": "math",
        "level": "profile",
        "year": 2026,
        "count": 500,
        "sha256Checksum": checksum,
        "cases": cases
    }
    with open("reference_dataset_500.json", "w", encoding="utf-8") as f:
        json.dump(ref_json, f, ensure_ascii=False, indent=2)

    return scale_json, task_stats_json, anomalies_json, ref_json


if __name__ == "__main__":
    scores_avg, probs_avg = simulate_exam_session(TASK_COMPLETION_RATE_AVG, n_students=500000, seed=42)
    scale_main = build_primary_to_secondary_scale(TASK_COMPLETION_RATE_AVG)
    validate_data_and_scale(scores_avg, scale_main)
    export_all_json_contracts(scores_avg, probs_avg, scale_main)
    print("Все расчёты и JSON-контракты успешно сформированы.")
