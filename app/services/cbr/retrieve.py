import os
import json
import math
import pandas as pd


def load_cbr_assets(base_dir: str) -> tuple[pd.DataFrame, dict, dict]:
    """
    Load semua aset yang dibutuhkan CBR:
    - dataset_cbr.csv  : basis kasus ternormalisasi
    - cbr_weights.json : bobot tiap fitur dari hasil RST
    - cbr_minmax.json  : parameter min-max untuk normalisasi input baru
    """
    dataset_path = os.path.join(base_dir, "data", "preprocessed", "dataset_cbr.csv")
    weights_path = os.path.join(base_dir, "model", "cbr_weights.json")
    minmax_path  = os.path.join(base_dir, "model", "cbr_minmax.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset_cbr.csv tidak ditemukan di: {dataset_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"cbr_weights.json tidak ditemukan di: {weights_path}")
    if not os.path.exists(minmax_path):
        raise FileNotFoundError(f"cbr_minmax.json tidak ditemukan di: {minmax_path}")

    df_cbr    = pd.read_csv(dataset_path)
    weights   = json.load(open(weights_path))
    minmax    = json.load(open(minmax_path))

    return df_cbr, weights, minmax


def normalize_input(raw_input: dict, minmax_params: dict) -> dict:
    """
    Normalisasi input user baru ke skala 0-1 menggunakan
    parameter min-max yang sama dengan saat training.

    Contoh:
        raw_input    = {"studytime": 2, "absences": 4, ...}
        minmax_params = {"studytime": {"min": 1, "max": 4}, ...}
        hasil        = {"studytime": 0.3333, "absences": 0.0526, ...}
    """
    normalized = {}
    for feature, params in minmax_params.items():
        value   = raw_input.get(feature, 0)
        min_val = params['min']
        max_val = params['max']

        if max_val == min_val:
            normalized[feature] = 0.0
        else:
            norm_value = (value - min_val) / (max_val - min_val)
            # Clamp ke rentang [0, 1] untuk antisipasi input di luar range training
            normalized[feature] = round(max(0.0, min(1.0, norm_value)), 4)

    return normalized


def weighted_euclidean_distance(input_norm: dict, case_row: dict, weights: dict) -> float:
    """
    Hitung jarak Weighted Euclidean Distance antara input baru
    dengan satu kasus di basis kasus.

    Rumus: sqrt( Σ weight[i] * (input[i] - case[i])² )

    Semakin kecil jarak → semakin mirip kasusnya.
    """
    total = 0.0
    for feature, weight in weights.items():
        input_val = input_norm.get(feature, 0.0)
        case_val  = case_row.get(feature, 0.0)
        total += weight * ((input_val - case_val) ** 2)

    return math.sqrt(total)


def distance_to_similarity(distance: float) -> float:
    """
    Konversi jarak ke nilai kemiripan (similarity) skala 0-1.

    Rumus: similarity = 1 / (1 + distance)
    - Jarak 0   → similarity 1.0 (identik)
    - Jarak besar → similarity mendekati 0
    """
    return round(1 / (1 + distance), 4)


def retrieve(raw_input: dict, base_dir: str = None) -> dict:
    """
    Fungsi utama CBR Retrieval.
    Mencari kasus paling mirip dari basis kasus untuk setiap kelas
    dan mengembalikan CBR Score per kelas.

    Returns:
        {
            "Normal":   0.87,  ← similarity tertinggi dari kasus berlabel Normal
            "Waspada":  0.61,
            "Bahaya":   0.45
        }
    """
    if base_dir is None:
        # Resolusi otomatis path dari lokasi file ini
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # --- Step 1: Load semua aset ---
    df_cbr, weights, minmax_params = load_cbr_assets(base_dir)

    # --- Step 2: Normalisasi input user ---
    input_normalized = normalize_input(raw_input, minmax_params)

    # --- Step 3: Hitung similarity ke setiap kasus ---
    labels    = ["Normal", "Waspada", "Bahaya"]
    best_similarity = {label: 0.0 for label in labels}

    for _, row in df_cbr.iterrows():
        case_label = row['decision']

        # Lewati kasus dengan label di luar yang dikenal
        if case_label not in labels:
            continue

        # Konversi baris ke dict (hanya fitur, tanpa kolom decision)
        case_features = {feature: row[feature] for feature in weights.keys() if feature in row}

        # Hitung jarak lalu konversi ke similarity
        distance   = weighted_euclidean_distance(input_normalized, case_features, weights)
        similarity = distance_to_similarity(distance)

        # Simpan hanya similarity tertinggi untuk setiap kelas
        if similarity > best_similarity[case_label]:
            best_similarity[case_label] = similarity

    return best_similarity