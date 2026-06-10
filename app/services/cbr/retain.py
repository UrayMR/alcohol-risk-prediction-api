import os
import json
import pandas as pd

def retain_new_case(raw_input: dict, final_decision: str, base_dir: str = "") -> bool:
    """
    Menyimpan kasus baru yang sudah diputuskan ke dalam basis kasus CBR
    (dataset_cbr.csv) agar sistem semakin kaya pengalaman seiring waktu.

    Proses ini adalah bagian dari siklus CBR:
    Retrieve → Reuse → Revise → Retain ← di sinilah fungsi ini berperan
    """
    if base_dir is None:
        # Menyesuaikan path agar mengarah ke root project secara presisi
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    dataset_path = os.path.join(base_dir, "data", "preprocessed", "dataset_cbr.csv")
    minmax_path  = os.path.join(base_dir, "model", "cbr_minmax.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset_cbr.csv tidak ditemukan di: {dataset_path}")
    if not os.path.exists(minmax_path):
        raise FileNotFoundError(f"cbr_minmax.json tidak ditemukan di: {minmax_path}")

    try:
        # 1. Load parameter Min-Max dari hasil fase offline
        with open(minmax_path, 'r', encoding='utf-8') as f:
            minmax_params = json.load(f)

        # 2. Normalisasi input baru menggunakan parameter min-max yang konsisten
        new_case = {}
        for feature, params in minmax_params.items():
            value   = raw_input.get(feature, 0)
            min_val = params['min']
            max_val = params['max']

            if max_val == min_val:
                new_case[feature] = 0.0
            else:
                norm_value = (value - min_val) / (max_val - min_val)
                new_case[feature] = round(max(0.0, min(1.0, norm_value)), 4)

        # Sesuaikan key kolom target keputusan dengan header asli dataset kelompokmu
        # Di sini kita asumsikan namanya 'decision' sesuai kodemu
        new_case['decision'] = final_decision

        # 3. Baca dataset CBR lama
        df_cbr = pd.read_csv(dataset_path)

        # 4. Ubah data baru menjadi DataFrame
        new_row = pd.DataFrame([new_case])

        # Ini mencegah kolom tertukar posisi saat proses append (pd.concat)
        new_row = new_row[df_cbr.columns]

        # 5. Gabungkan dan simpan kembali ke CSV
        df_updated = pd.concat([df_cbr, new_row], ignore_index=True)
        df_updated.to_csv(dataset_path, index=False)

        print(f"Kasus baru berhasil disimpan ke basis kasus CBR. Total kasus: {len(df_updated)}")
        return True

    except Exception as e:
        print(f"Gagal menyimpan kasus baru: {e}")
        return False