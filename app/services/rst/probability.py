ALL_CLASSES = ["Normal", "Waspada", "Bahaya"]


def calculate_class_probability(matched_rules: list) -> dict:
    """
    Menghitung probabilitas tiap kelas keputusan dari rules yang cocok.

    - Positive Region (confidence == 1.0):
      Probabilitas kelas yang cocok = 1.0, kelas lain = 0.0
      Contoh: {"Normal": 1.0, "Waspada": 0.0, "Bahaya": 0.0}

    - Boundary Region (confidence < 1.0):
      Probabilitas kelas utama = confidence rule.
      Sisa (1 - confidence) dibagi rata ke kelas lain.
      Contoh rule Normal confidence 0.61:
      {"Normal": 0.61, "Waspada": 0.195, "Bahaya": 0.195}

    - Empty Region (tidak ada rule cocok):
      Probabilitas semua kelas = 0.0
      Contoh: {"Normal": 0.0, "Waspada": 0.0, "Bahaya": 0.0}

    Parameter:
        matched_rules (list): List rules yang cocok dari rule_matcher.

    Returns:
        dict: Probabilitas tiap kelas.
    """
    # Empty region
    if not matched_rules:
        return {cls: 0.0 for cls in ALL_CLASSES}

    rule = matched_rules[0]
    confidence = rule.get("confidence", 1.0)
    main_class = rule["then"]

    # Positive region - confidence 1.0
    if confidence == 1.0:
        return {cls: (1.0 if cls == main_class else 0.0) for cls in ALL_CLASSES}

    # Boundary region - confidence < 1.0
    # Sisa confidence dibagi rata ke kelas lain
    other_classes = [cls for cls in ALL_CLASSES if cls != main_class]
    remaining = round(1.0 - confidence, 4)
    prob_per_other = round(remaining / len(other_classes), 4)

    probabilities = {}
    for cls in ALL_CLASSES:
        if cls == main_class:
            probabilities[cls] = round(confidence, 4)
        else:
            probabilities[cls] = prob_per_other

    return probabilities


def get_final_prediction(probabilities: dict) -> str:
    """
    Mengambil kelas dengan probabilitas tertinggi sebagai hasil prediksi akhir.

    Parameter:
        probabilities (dict): Hasil dari calculate_class_probability.

    Returns:
        str: Label kelas dengan probabilitas tertinggi.
    """
    if not probabilities:
        return "Tidak Diketahui"

    return max(probabilities, key=probabilities.get)