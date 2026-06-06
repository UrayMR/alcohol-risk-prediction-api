import os
import json
import pandas as pd
from training.utils.alcohol_decision import calculate_alcohol_decision


def load_raw_data(directory_path: str) -> pd.DataFrame:
    csv_path = os.path.join(directory_path, "dataset.csv")
    xlsx_path = os.path.join(directory_path, "dataset.xlsx")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(
            "Dataset tidak ditemukan di folder data/raw/. "
            "Pastikan bernama dataset.csv atau dataset.xlsx"
        )


def normalize_for_cbr(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalisasi fitur kondisi ke skala 0-1 menggunakan Min-Max Normalization.
    Rumus: x_norm = (x - x_min) / (x_max - x_min)

    Selain dataset ternormalisasi, fungsi ini juga mengembalikan
    minmax_params (nilai min & max tiap fitur) agar bisa digunakan
    saat normalisasi input user baru di fase online.
    """
    features = ['studytime', 'absences', 'failures', 'famrel', 'freetime', 'goout']
    cbr_df = pd.DataFrame()
    minmax_params = {}

    for feature in features:
        if feature not in df.columns:
            raise KeyError(f"Kolom '{feature}' tidak ditemukan di dataset!")

        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        minmax_params[feature] = {"min": min_val, "max": max_val}

        # Hindari pembagian dengan nol jika semua nilai sama
        if max_val == min_val:
            cbr_df[feature] = 0.0
        else:
            cbr_df[feature] = (df[feature] - min_val) / (max_val - min_val)

        # Bulatkan ke 4 desimal agar file CSV lebih rapi
        cbr_df[feature] = cbr_df[feature].round(4)

    # Tambahkan kolom label keputusan (solusi kasus)
    cbr_df['decision'] = calculate_alcohol_decision(df, dalc_col='Dalc', walc_col='Walc')

    return cbr_df, minmax_params


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw")
    preprocessed_dir = os.path.join(base_dir, "data", "preprocessed")
    model_dir = os.path.join(base_dir, "model")

    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    try:
        df_raw = load_raw_data(raw_dir)
        print("Mulai preprocessing CBR (Normalisasi Min-Max)...")

        df_cbr, minmax_params = normalize_for_cbr(df_raw)

        # Simpan dataset ternormalisasi
        output_csv_path = os.path.join(preprocessed_dir, "dataset_cbr.csv")
        df_cbr.to_csv(output_csv_path, index=False)
        print(f"Sukses! Dataset CBR disimpan di: {output_csv_path}")

        # Simpan parameter min-max untuk digunakan di fase online
        output_minmax_path = os.path.join(model_dir, "cbr_minmax.json")
        with open(output_minmax_path, 'w') as f:
            json.dump(minmax_params, f, indent=4)
        print(f"Sukses! Parameter Min-Max disimpan di: {output_minmax_path}")

        print("\nRingkasan parameter Min-Max:")
        for feature, params in minmax_params.items():
            print(f"  {feature}: min={params['min']}, max={params['max']}")

    except Exception as e:
        print(f"Error pada CBR Preprocessing: {e}")


if __name__ == "__main__":
    main()