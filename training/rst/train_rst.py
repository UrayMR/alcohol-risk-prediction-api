import os
import json
import pandas as pd
import numpy as np

def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data preprocessing tidak ditemukan di: {file_path}")
    return pd.read_csv(file_path)

def get_indiscernibility_classes(df: pd.DataFrame, attributes: list) -> list:
    """
    Mencari himpunan ekuivalen / kelas indiscernibility berdasarkan sekumpulan atribut yang dipilih.
    Mengelompokkan baris data siswa yang memiliki nilai atribut kondisi SAMA PERSIS.
    
    Contoh Hasil Output:
    [
        [0, 2],  Kelompok 1: Berisi Siswa Indeks 0 & Indeks 2 (Keduanya punya data: studytime='Rendah', absences='Sedikit', failures='Tidak', famrel='Baik', freetime='Sedang', goout='Jarang')
        [1, 4],  Kelompok 2: Berisi Siswa Indeks 1 & Indeks 4 (Keduanya punya data: studytime='Tinggi', absences='Banyak', failures='Banyak', famrel='Buruk', freetime='Rendah', goout='Sering')
        [3]      Kelompok 3: Berisi Siswa Indeks 3 (Sendirian, tidak ada siswa lain yang kombinasi fiturnya sama persis dengan dia)
    ]
    """
    # Mengelompokkan baris berdasarkan kesamaan nilai atribut
    grouped = df.groupby(attributes).groups

    # Mengembalikan daftar indeks baris untuk setiap kelompok
    return [list(indices) for indices in grouped.values()]


def calculate_positive_region(df: pd.DataFrame, condition_attributes: list, decision_attributes: str) -> list:
    """
    Menghitung Positive Region (Himpunan dari kelompok siswa yang data kondisinya 
    sepenuhnya menghasilkan keputusan/label akhir yang KONSISTEN di dataset).
    """

    # Ambil kelas ekuivalen berdasarkan atribut kondisi siswa
    condition_classes = get_indiscernibility_classes(df, condition_attributes)
    # Contoh condition_classes:
    # [
    #    [0, 2],  Kelompok Siswa A (Kombinasi data kondisi sama)
    #    [1, 4],  Kelompok Siswa B (Kombinasi data kondisi sama)
    #    [3, 5]   Kelompok Siswa C (Kombinasi data kondisi sama)
    # ]

    # Ambil kelas ekuivalen berdasarkan kolom keputusan (Hanya melihat label: Normal, Waspada, Bahaya)
    decision_classes = get_indiscernibility_classes(df, [decision_attributes])
    # Contoh decision_classes:
    # [
    #    [0, 2, 3, 5], Kelompok Keputusan "Normal" (Semua siswa yang hasil akhirnya Normal)
    #    [1, 4]        Kelompok Keputusan "Waspada" (Semua siswa yang hasil akhirnya Waspada)
    # ]
    
    positive_region = []
    
    # Looping untuk mengecek setiap kelompok kondisi (condition_classes) apakah dia sepenuhnya masuk ke dalam salah satu kelompok keputusan (decision_classes)
    # Simulasi:
    # Kasus 1: Mengecek Kelompok Siswa A [0, 2]
    # Apakah [0, 2] ada di dalam Kelompok "Normal" [0, 2, 3, 5]? YA (masuk Positive Region)
    # Kasus 2: Mengecek Kelompok Siswa C [3, 5]
    # Apakah [3, 5] ada di dalam Kelompok "Normal" [0, 2, 3, 5]? YA (masuk Positive Region)
    # Kasus Inkonsisten (Boundary Region):
    # Misal ada Kelompok Siswa D berisi indeks [6, 7]. Siswa 6 hasil akhirnya "Normal", Siswa 7 hasil akhirnya "Waspada".
    # Saat dicek, [6, 7] tidak akan menjadi subset utuh dari kelompok keputusan manapun karena hasilnya pecah/abu-abu.
    # Kelompok seperti ini otomatis ditolak dan TIDAK MASUK ke Positive Region.
    for c_class in condition_classes:
        is_subset = False # Flag untuk menandai apakah kelompok kondisi ini sepenuhnya masuk ke dalam salah satu kelompok keputusan
        for d_class in decision_classes:
            # Cek apakah kelompok condition ini masuk ke dalam kelompok decision
            if set(c_class).issubset(set(d_class)):
                is_subset = True # Set flag true
                break
        if is_subset:
            positive_region.extend(c_class) # Tambahkan kelompok kondisi ke Positive Region
            
    return positive_region

def calculate_dependency_degree(df: pd.DataFrame, condition_attributes: list, decision_attributes: str) -> float:
    """
    Menghitung Derajat Ketergantungan antara tiap atribut kondisi dengan atribut keputusan.
    Derajat Ketergantungan = Jumlah di Positive region / Total jumlah data
    """

    # Fallback jika tidak ada atribut kondisi, maka derajat ketergantungan dianggap 0
    if not condition_attributes:
        return 0.0
    
    # Ambil list positive region berdasarkan atribut kondisi dan keputusan
    pos_region = calculate_positive_region(df, condition_attributes, decision_attributes)

    # Ambil total jumlah data di positive region
    total_pos_region = len(pos_region)

    # Ambil total jumlah data
    total_universe = len(df)

    return total_pos_region / total_universe

def calculate_attribute_significance(df: pd.DataFrame, condition_attributes: list, decision_attributes: str) -> dict:
    """
    Menghitung signifikansi setiap atribut dengan cara membandingkan derajat ketergantungan total dengan derajat ketergantungan setelah menghilangkan atribut tersebut.
    Semakin besar selisihnya, semakin signifikan atribut tersebut untuk keputusan akhir.
    """
    all_dependency_degree = calculate_dependency_degree(df, condition_attributes, decision_attributes)
    significance = {}
    
    for attribute in condition_attributes:
        # Buat daftar atribut kondisi yang dikurangi satu atribut (attribute) untuk menghitung derajat ketergantungan tanpa atribut tersebut
        reduced_attrs = [diff_attribute for diff_attribute in condition_attributes if diff_attribute != attribute]

        # Hitung derajat ketergantungan tanpa atribut tersebut
        reduced_dependency_degree = calculate_dependency_degree(df, reduced_attrs, decision_attributes)
        
        # Hitung selisihnya sebagai ukuran signifikansi
        sig_value = all_dependency_degree - reduced_dependency_degree

        # Jika nilai minus/0 karena pembulatan atau data, beri batas bawah kecil (misal 0.01) agar CBR tetap punya bobot
        significance[attribute] = max(round(sig_value, 4), 0.01)
        
    return significance

def generate_rules_extraction(df: pd.DataFrame, condition_attrs: list, decision_attr: str) -> list:
    """
    Mengekstrak rules unik (IF-THEN) dari dataset berdasarkan atribut kondisi yang dipilih dan atribut keputusan. 
    """

    # Ambil baris unik berdasarkan atribut kondisi yang dipilih
    unique_cases = df.drop_duplicates(subset=condition_attrs)
    rules = []
    
    # Looping untuk setiap baris unik yang didapatkan
    for _, row in unique_cases.iterrows():
        # Membentuk bagian IF (Kondisi)
        if_part = {attr: row[attr] for attr in condition_attrs}
        # Membentuk bagian THEN (Keputusan)
        then_part = row[decision_attr]
        
        # Contoh Hasil dari Query: "`studytime` == 'Rendah' & `absences` == 'Sedikit' & `failures` == 'Tidak' & `famrel` == 'Baik' & `freetime` == 'Sedang' & `goout` == 'Jarang'"
        query = " & ".join([f"`{k}` == '{v}'" for k, v in if_part.items()])

        # Query dataset untuk mencari semua baris yang cocok dengan rules ini
        matching_df = df.query(query)

        # Hitung berapa banyak baris yang cocok dengan rules ini
        total_match = len(matching_df)

        # Hitung berapa banyak baris yang cocok dengan rules ini DAN hasil keputusannya sesuai dengan THEN part (decision)
        correct_match = len(matching_df[matching_df[decision_attr] == then_part])
        
        # Hitung confidence level (probabilitas keputusan benar)
        # Sebagai fallback, jika total_match = 0 (tidak ada data yang cocok dengan rules ini), maka confidence dianggap 0.0 untuk menghindari pembagian dengan nol.
        # Skor Confidence Rules = Jumlah baris yang cocok dengan rules ini dan hasil keputusannya benar / Jumlah baris yang cocok dengan rules ini
        confidence = 0.0
        if total_match > 0:
            confidence = correct_match / total_match 
        
        rules.append({
            "if": if_part,
            "then": then_part,
            "confidence": round(confidence, 2)
        })
        
    return rules

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rst_data_path = os.path.join(base_dir, "data", "preprocessed", "dataset_rst.csv")
    output_rules_path = os.path.join(base_dir, "model", "rst_rules.json")
    output_weights_path = os.path.join(base_dir, "model", "cbr_weights.json")
    
    print("Memulai Proses Training RST")
    try:
        df = load_preprocessed_data(rst_data_path)
        
        # Tentukan kolom kondisi dan keputusan
        decision_col = 'decision'
        condition_cols = [col for col in df.columns if col != decision_col]
        
        print(f"Atribut Kondisi (C): {condition_cols}")
        print(f"Atribut Keputusan (D): {decision_col}")
        
        total_dependency_degree = calculate_dependency_degree(df, condition_cols, decision_col)
        print(f"Total Dependency Degree: {round(total_dependency_degree, 4)}")
        
        print("\nMenghitung signifikansi atribut untuk bobot CBR...")
        weights = calculate_attribute_significance(df, condition_cols, decision_col)
        
        # Normalisasi bobot agar total jumlah bobotnya = 1.0 (Opsional)
        total_weight = sum(weights.values())
        normalized_weights = {attr_name: round(attr_weight / total_weight, 4) for attr_name, attr_weight in weights.items()}
        
        print("Hasil Bobot Variasi RST untuk CBR:")
        for attr_name, attr_weight in normalized_weights.items():
            print(f" - {attr_name}: {attr_weight} (Signifikansi Asli: {weights[attr_name]})")
            
        print("\nMengekstrak Aturan keputusan (Rules)...")
        extracted_rules = generate_rules_extraction(df, condition_cols, decision_col)
        print(f"Berhasil mengekstrak {len(extracted_rules)} aturan.")
        
        # Simpan aturan ke model/rst_rules.json
        os.makedirs(os.path.dirname(output_rules_path), exist_ok=True)
        with open(output_rules_path, 'w') as f:
            json.dump(extracted_rules, f, indent=4)

        # Simpan bobot ke model/cbr_weights.json
        os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
        with open(output_weights_path, 'w') as f:
            json.dump(normalized_weights, f, indent=4)

        print(f"\nSukses! Aturan disimpan di: {output_rules_path}")
        print(f"Bobot disimpan di: {output_weights_path}")
        
    except Exception as e:
        print(f"Error saat training RST: {e}")

if __name__ == "__main__":
    main()