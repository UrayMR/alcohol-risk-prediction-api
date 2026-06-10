import json
import os
from services.rst.discretizer import discretize_input
from services.rst.rule_matcher import match_rules, get_region_type
from services.rst.probability import calculate_class_probability

# Path ke file model hasil training
# Perlu diganti jadi:
# Lebih aman, tidak perlu hitung naik berapa level
# Naik 3x: rst → services → app → alcohol-risk-prediction-api (ROOT)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RST_RULES_PATH = os.path.join(BASE_DIR, "model", "rst_rules.json")


def load_rst_rules() -> list:
    """Membaca rules hasil training dari rst_rules.json."""
    with open(RST_RULES_PATH, 'r') as f:
        return json.load(f)


def predict_rst(input_data: dict) -> dict:
    """
    Fungsi utama prediksi RST. Dipanggil langsung dari route FastAPI.

    Alur:
        1. Diskretisasi input mentah dari frontend
        2. Load rules dari model
        3. Cocokkan input ke rules
        4. Tentukan tipe region (positive / boundary / empty)
        5. Hitung probabilitas tiap kelas
        6. Return hasil prediksi lengkap

    Parameter:
        input_data (dict): Data mentah dari frontend (JSON).
        Contoh:
        {
            "studytime": 2,
            "absences": 3,
            "failures": 0,
            "famrel": 4,
            "freetime": 3,
            "goout": 2
        }

    Returns:
        dict: Probabilitas tiap kelas per region, diteruskan ke CBR untuk integrasi.
        Contoh (Positive Region):
        {
            "status": "success",
            "region": "positive",
            "probabilities": {"Normal": 1.0},
            "discretized_input": {"studytime": "Rendah", ...}
        }

        Contoh (Boundary Region):
        {
            "status": "success",
            "region": "boundary",
            "probabilities": {"Normal": 0.28, "Waspada": 0.72},
            "discretized_input": {...}
        }

        Contoh (Empty Region):
        {
            "status": "success",
            "region": "empty",
            "probabilities": {},
            "discretized_input": {...}
        }
    """
    # Step 1: Diskretisasi input
    discretized = discretize_input(input_data)

    # Step 2: Load rules dari model
    rules = load_rst_rules()

    # Step 3: Cocokkan input ke rules
    matched = match_rules(discretized, rules)

    # Step 4: Tentukan tipe region
    region = get_region_type(matched)

    # Step 5: Hitung probabilitas
    probabilities = calculate_class_probability(matched)

    return {
        "region": region,
        "probabilities": probabilities,
        "discretized_input": discretized
    }