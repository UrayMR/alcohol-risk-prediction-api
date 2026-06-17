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
    """
    # FIX WARNING: Jika hanya ada 1 atribut di dalam list, gunakan string-nya saja (scalar)
    # Jika ada lebih dari 1, tetap gunakan list tersebut.
    groupby_param = attributes[0] if len(attributes) == 1 else attributes

    # Mengelompokkan baris berdasarkan kesamaan nilai atribut
    grouped = df.groupby(groupby_param).groups

    # Mengembalikan daftar indeks baris untuk setiap kelompok
    return [list(indices) for indices in grouped.values()]


def calculate_positive_region(df: pd.DataFrame, condition_attributes: list, decision_attributes: str) -> list:
    """
    Menghitung Positive Region (Himpunan dari kelompok siswa yang data kondisinya 
    sepenuhnya menghasilkan keputusan/label akhir yang KONSISTEN di dataset).
    """

    # Ambil kelas ekuivalen berdasarkan atribut kondisi siswa
    condition_classes = get_indiscernibility_classes(df, condition_attributes)

    # Ambil kelas ekuivalen berdasarkan kolom keputusan
    # Catatan: Kita tetap mengirimkan list berisi 1 elemen, namun di dalam fungsi 
    # get_indiscernibility_classes sudah otomatis ditangani agar tidak memicu warning.
    decision_classes = get_indiscernibility_classes(df, [decision_attributes])
    
    positive_region = []
    
    for c_class in condition_classes:
        is_subset = False 
        for d_class in decision_classes:
            # Cek apakah kelompok condition ini masuk ke dalam kelompok decision
            if set(c_class).issubset(set(d_class)):
                is_subset = True 
                break
        if is_subset:
            positive_region.extend(c_class) 
            
    return positive_region

def find_reduct(df: pd.DataFrame, condition_attributes: list, decision_attributes: str) -> list:
    """
    Mencari Reduct dengan pendekatan Greedy (QuickReduct) yang diperbaiki.
    Dilengkapi cetakan log proses kombinasi dan perhitungan di terminal.
    """
    max_dependency = calculate_dependency_degree(df, condition_attributes, decision_attributes)
    reduct = []
    
    print(f"-> Target Dependency Maksimal (Semua Atribut): {round(max_dependency, 4)}")
    
    iteration = 1
    while True:
        print(f"\n[ ITERASI {iteration} ]")
        best_attr = None
        
        # Skor dependency saat ini sebelum ditambah atribut baru
        current_base_dep = calculate_dependency_degree(df, reduct, decision_attributes) if reduct else 0.0
        print(f"Kondisi Reduct Saat Ini: {reduct} (Dependency: {round(current_base_dep, 4)})")
        print("Mengecek kombinasi atribut baru:")
        
        # Set batas bawah pencarian
        best_dependency = calculate_dependency_degree(df, reduct, decision_attributes) if reduct else -1.0
        
        for attr in condition_attributes:
            if attr not in reduct:
                current_reduct = reduct + [attr]
                current_dependency = calculate_dependency_degree(df, current_reduct, decision_attributes)
                
                # Cetak skor kombinasi yang sedang diuji
                print(f"   - Mencoba: {current_reduct} -> Dependency: {round(current_dependency, 4)}")
                
                if current_dependency >= best_dependency:
                    best_dependency = current_dependency
                    best_attr = attr
                    
        if best_attr is not None:
            reduct.append(best_attr)
            print(f"\n>> Atribut Terpilih pada Iterasi {iteration}: '{best_attr}'")
            print(f">> Nilai Dependency Baru: {round(best_dependency, 4)}")
            
            # Jika sudah mencapai atau mendekati nilai maksimal dependency, hentikan.
            if np.isclose(best_dependency, max_dependency, atol=1e-4):
                print(f"\n[SUKSES] Target dependency maksimal ({round(max_dependency, 4)}) TELAH TERCAPAI!")
                break
        else:
            print("\n[STOP] Tidak ada lagi atribut yang menaikkan atau mempertahankan dependency.")
            break
            
        iteration += 1
            
    # Fallback jika kosong
    if not reduct:
        print("\n[WARNING] Reduct kosong, mengembalikan seluruh atribut awal sebagai fallback.")
        reduct = condition_attributes
        
    return reduct

def calculate_dependency_degree(df: pd.DataFrame, condition_attributes: list, decision_attributes: str) -> float:
    """
    Menghitung Derajat Ketergantungan antara tiap atribut kondisi dengan atribut keputusan.
    """
    if not condition_attributes:
        return 0.0
    
    pos_region = calculate_positive_region(df, condition_attributes, decision_attributes)
    total_pos_region = len(pos_region)
    total_universe = len(df)

    return total_pos_region / total_universe

def calculate_attribute_significance(df: pd.DataFrame, condition_attributes: list, decision_attributes: str) -> dict:
    """
    Menghitung signifikansi setiap atribut.
    """
    all_dependency_degree = calculate_dependency_degree(df, condition_attributes, decision_attributes)
    significance = {}
    
    for attribute in condition_attributes:
        reduced_attrs = [diff_attribute for diff_attribute in condition_attributes if diff_attribute != attribute]
        reduced_dependency_degree = calculate_dependency_degree(df, reduced_attrs, decision_attributes)
        
        sig_value = all_dependency_degree - reduced_dependency_degree
        significance[attribute] = max(round(sig_value, 4), 0.01)
        
    return significance

def generate_rules_extraction(df: pd.DataFrame, condition_attrs: list, decision_attr: str) -> list:
    """
    Mengekstrak rules unik (IF-THEN) dari dataset. 
    """
    unique_cases = df.drop_duplicates(subset=condition_attrs)
    rules = []
    
    for _, row in unique_cases.iterrows():
        if_part = {attr: row[attr] for attr in condition_attrs}
        then_part = row[decision_attr]
        
        query = " & ".join([f"`{k}` == '{v}'" for k, v in if_part.items()])
        matching_df = df.query(query)
        total_match = len(matching_df)
        correct_match = len(matching_df[matching_df[decision_attr] == then_part])
        
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
        
        decision_col = 'decision'
        condition_cols = [col for col in df.columns if col != decision_col]
        
        print(f"Atribut Kondisi (C): {condition_cols}")
        print(f"Atribut Keputusan (D): {decision_col}")
        
        total_dependency_degree = calculate_dependency_degree(df, condition_cols, decision_col)
        print(f"Total Dependency Degree: {round(total_dependency_degree, 4)}")
        
        print("\nMenghitung signifikansi atribut untuk bobot CBR...")
        weights = calculate_attribute_significance(df, condition_cols, decision_col)
        
        total_weight = sum(weights.values())
        normalized_weights = {attr_name: round(attr_weight / total_weight, 4) for attr_name, attr_weight in weights.items()}
        
        print("Hasil Bobot Variasi RST untuk CBR:")
        for attr_name, attr_weight in normalized_weights.items():
            print(f" - {attr_name}: {attr_weight} (Signifikansi Asli: {weights[attr_name]})")
            
        print("\nMencari Atribut Reduksi (Reduct)...")
        important_condition_cols = find_reduct(df, condition_cols, decision_col)
        print(f"\nAtribut hasil reduksi RST: {important_condition_cols}")

        print("\nMengekstrak Aturan keputusan (Rules)...")
        extracted_rules = generate_rules_extraction(df, important_condition_cols, decision_col)
        print(f"Berhasil mengekstrak {len(extracted_rules)} aturan.")
        
        os.makedirs(os.path.dirname(output_rules_path), exist_ok=True)
        with open(output_rules_path, 'w') as f:
            json.dump(extracted_rules, f, indent=4)

        os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
        with open(output_weights_path, 'w') as f:
            json.dump(normalized_weights, f, indent=4)

        print(f"\nSukses! Aturan disimpan di: {output_rules_path}")
        print(f"Bobot disimpan di: {output_weights_path}")
        
    except Exception as e:
        print(f"Error saat training RST: {e}")

if __name__ == "__main__":
    main()