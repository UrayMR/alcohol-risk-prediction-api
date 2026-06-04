import pandas as pd

def calculate_alcohol_decision(df: pd.DataFrame, dalc_col: str = 'DALC', walc_col: str = 'WALC') -> pd.Series:
    """
    Menghitung kolom decision kecanduan alkohol:
    Score = (0.6 * DALC) + (0.4 * WALC)
    Score <= 2.0   -> Normal
    Score <= 3.5   -> Waspada
    Score > 3.5    -> Bahaya
    """
    if dalc_col not in df.columns or walc_col not in df.columns:
        raise KeyError(f"Kolom {dalc_col} atau {walc_col} tidak ditemukan di dataset!")

    weighted_score = (0.6 * df[dalc_col]) + (0.4 * df[walc_col])
    
    decision = pd.Series("Bahaya", index=df.index)
    
    decision[weighted_score <= 3.5] = "Waspada"
    decision[weighted_score <= 2.0] = "Normal"
    
    return decision