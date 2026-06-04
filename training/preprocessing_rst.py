import os
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
        raise FileNotFoundError("Dataset tidak ditemukan di folder data/raw/. Pastikan bernama dataset.csv atau dataset.xlsx")

def discretize_for_rst(df: pd.DataFrame) -> pd.DataFrame:
    rst_df = pd.DataFrame()
    
    # 1. Studytime (1-2: Rendah, 3: Sedang, 4: Tinggi)
    if 'studytime' in df.columns:
        rst_df['studytime'] = pd.cut(df['studytime'], bins=[0, 2, 3, 4], labels=['Rendah', 'Sedang', 'Tinggi'])
        
    # 2. Absences (0-5: Sedikit, 6-15: Sedang, >15: Banyak)
    if 'absences' in df.columns:
        rst_df['absences'] = pd.cut(df['absences'], bins=[-1, 5, 15, float('inf')], labels=['Sedikit', 'Sedang', 'Banyak'])
        
    # 3. Failures (0: Tidak, 1-2: Sedikit, >=3: Banyak)
    if 'failures' in df.columns:
        rst_df['failures'] = pd.cut(df['failures'], bins=[-1, 0, 2, float('inf')], labels=['Tidak', 'Sedikit', 'Banyak'])
        
    # 4. Famrel (1-2: Buruk, 3-4: Baik, 5: Sangat Baik)
    if 'famrel' in df.columns:
        rst_df['famrel'] = pd.cut(df['famrel'], bins=[0, 2, 4, 5], labels=['Buruk', 'Baik', 'Sangat Baik'])
        
    # 5. Freetime (1-2: Rendah, 3-4: Sedang, 5: Tinggi)
    if 'freetime' in df.columns:
        rst_df['freetime'] = pd.cut(df['freetime'], bins=[0, 2, 4, 5], labels=['Rendah', 'Sedang', 'Tinggi'])
        
    # 6. Goout (1-2: Jarang, 3-4: Sedang, 5: Sering)
    if 'goout' in df.columns:
        rst_df['goout'] = pd.cut(df['goout'], bins=[0, 2, 4, 5], labels=['Jarang', 'Sedang', 'Sering'])
        
    # Decision untuk alkohol berdasarkan DALC dan WALC
    rst_df['decision'] = calculate_alcohol_decision(df, dalc_col='DALC', walc_col='WALC')

    return rst_df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw")
    preprocessed_dir = os.path.join(base_dir, "data", "preprocessed")
    
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    try:
        df_raw = load_raw_data(raw_dir)
        print("Mulai preprocessing RST (Diskretisasi)...")
        df_rst = discretize_for_rst(df_raw)
        
        output_path = os.path.join(preprocessed_dir, "dataset_rst.csv")
        df_rst.to_csv(output_path, index=False)
        print(f"Sukses! Data kategori RST disimpan di: {output_path}")
    except Exception as e:
        print(f"Error pada RST Preprocessing: {e}")

if __name__ == "__main__":
    main()