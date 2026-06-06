def match_rules(discretized_input: dict, rules: list) -> list:
    """
    Mencocokkan data input yang sudah didiskretisasi dengan rules hasil training RST.

    Parameter:
        discretized_input (dict): Data input yang sudah didiskretisasi.
        rules (list): List rules hasil training dari rst_rules.json.

    Returns:
        list: List rules yang cocok dengan input.
    """
    matched_rules = []

    for rule in rules:
        rule_conditions = rule.get("if", {})
        is_match = True

        # Cek apakah semua atribut kondisi di rule cocok dengan input
        for attr, value in rule_conditions.items():
            if attr not in discretized_input or discretized_input[attr] != value:
                is_match = False
                break

        if is_match:
            matched_rules.append(rule)

    return matched_rules


def get_region_type(matched_rules: list) -> str:
    """
    Menentukan tipe region berdasarkan confidence rules yang cocok.

    Returns:
        str:
            - "positive"  → semua rules yang cocok memiliki confidence == 1.0 (pasti)
            - "boundary"  → ada rule yang cocok dengan confidence < 1.0 (ambigu)
            - "empty"     → tidak ada rule yang cocok
    """
    if not matched_rules:
        return "empty"

    # Jika semua rules confidence 1.0 → positive (pasti)
    # Jika ada yang confidence < 1.0 → boundary (ambigu)
    all_certain = all(rule.get("confidence", 1.0) == 1.0 for rule in matched_rules)

    if all_certain:
        return "positive"
    else:
        return "boundary"