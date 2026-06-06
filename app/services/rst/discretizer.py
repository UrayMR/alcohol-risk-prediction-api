def discretize_input(input_data: dict) -> dict:
    """
    Mengubah data input mentah (numerik) dari frontend
    menjadi kategori diskrit yang sesuai dengan format training RST.

    Parameter:
        input_data (dict): Data mentah dari frontend dalam bentuk JSON.
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
        dict: Data yang sudah didiskretisasi.
        Contoh:
        {
            "studytime": "Rendah",
            "absences": "Sedikit",
            "failures": "Tidak",
            "famrel": "Baik",
            "freetime": "Sedang",
            "goout": "Sedang"
        }
    """
    discretized = {}

    # 1. Studytime (1-2: Rendah, 3: Sedang, 4: Tinggi)
    if 'studytime' in input_data:
        val = input_data['studytime']
        if val <= 2:
            discretized['studytime'] = 'Rendah'
        elif val == 3:
            discretized['studytime'] = 'Sedang'
        else:
            discretized['studytime'] = 'Tinggi'

    # 2. Absences (0-5: Sedikit, 6-15: Sedang, >15: Banyak)
    if 'absences' in input_data:
        val = input_data['absences']
        if val <= 5:
            discretized['absences'] = 'Sedikit'
        elif val <= 15:
            discretized['absences'] = 'Sedang'
        else:
            discretized['absences'] = 'Banyak'

    # 3. Failures (0: Tidak, 1-2: Sedikit, >=3: Banyak)
    if 'failures' in input_data:
        val = input_data['failures']
        if val == 0:
            discretized['failures'] = 'Tidak'
        elif val <= 2:
            discretized['failures'] = 'Sedikit'
        else:
            discretized['failures'] = 'Banyak'

    # 4. Famrel (1-2: Buruk, 3-4: Baik, 5: Sangat Baik)
    if 'famrel' in input_data:
        val = input_data['famrel']
        if val <= 2:
            discretized['famrel'] = 'Buruk'
        elif val <= 4:
            discretized['famrel'] = 'Baik'
        else:
            discretized['famrel'] = 'Sangat Baik'

    # 5. Freetime (1-2: Rendah, 3-4: Sedang, 5: Tinggi)
    if 'freetime' in input_data:
        val = input_data['freetime']
        if val <= 2:
            discretized['freetime'] = 'Rendah'
        elif val <= 4:
            discretized['freetime'] = 'Sedang'
        else:
            discretized['freetime'] = 'Tinggi'

    # 6. Goout (1-2: Jarang, 3-4: Sedang, 5: Sering)
    if 'goout' in input_data:
        val = input_data['goout']
        if val <= 2:
            discretized['goout'] = 'Jarang'
        elif val <= 4:
            discretized['goout'] = 'Sedang'
        else:
            discretized['goout'] = 'Sering'

    return discretized