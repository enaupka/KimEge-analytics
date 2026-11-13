import numpy as np
import pandas as pd


def validate_data_quality(scores_matrix: np.ndarray) -> bool:
    scores = np.asarray(scores_matrix)

    # 1. Проверка размерности (12 заданий профильной части)
    if scores.ndim != 2 or scores.shape[1] != 12:
        raise ValueError(f"Ошибка структуры: матрица должна иметь размерность (N, 12). Получено: {scores.shape}")

    # 2. Проверка на пропуски (NaN)
    if np.isnan(scores).any():
        nan_count = np.isnan(scores).sum()
        raise ValueError(f"Нарушение полноты данных (NFR-03): обнаружено {nan_count} пропусков (NaN).")

    # 3. Проверка на допустимые значения (только 0 или 1)
    if not np.isin(scores, [0, 1]).all():
        raise ValueError("Нарушение согласованности: матрица ответов содержит значения вне диапазона {0, 1}.")

    print("[DQ OK] Контроль качества данных успешно пройден: пропусков нет, формат бинарной матрицы корректен.")
    return True